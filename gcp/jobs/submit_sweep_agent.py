"""
submit_sweep_agent.py

Submits W&B sweep agent trials as either:
  - Vertex AI Custom Training Jobs  (--backend vertex,          default)
  - Compute Engine VM instances      (--backend compute-engine)

This is Workflow 1 — Hyperparameter Search.

How it works:
  1. This script auto-creates the W&B sweep (no manual step needed)
  2. Launches N parallel jobs/instances on the chosen backend
  3. Each job runs: wandb agent <sweep_id> --count 1
  4. W&B Bayesian optimizer picks next hyperparameters for each trial
  5. After all trials finish, fetch the best config automatically

Usage:
    # Vertex AI (default) — auto-create sweep + launch 10 trials
    python gcp/jobs/submit_sweep_agent.py --trials 10

    # Compute Engine — same sweep, VMs instead of Vertex AI
    python gcp/jobs/submit_sweep_agent.py --trials 10 --backend compute-engine

    # Resume an existing sweep on either backend
    python gcp/jobs/submit_sweep_agent.py --sweep-id <sweep_id> --trials 10

    # Dry-run — print job/instance config but do NOT submit
    python gcp/jobs/submit_sweep_agent.py --trials 3 --dry-run
"""

from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# On Windows gcloud is a .cmd file — shutil.which resolves the correct path
_GCLOUD = shutil.which("gcloud") or "gcloud"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PROJECT_ID, REGION, IMAGE_URI,
    TRAINER_SA,
    WANDB_PROJECT, WANDB_ENTITY,
    TRAIN_MACHINE_TYPE, TRAIN_ACCELERATOR, TRAIN_ACCEL_COUNT,
    CE_ZONE, CE_MACHINE_TYPE, CE_DISK_IMAGE_PROJECT, CE_DISK_IMAGE_FAMILY, CE_DISK_SIZE_GB,
)

_REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
_SWEEP_CONFIG = _REPO_ROOT / "model_2_training/configs/sweep/wandb_sweep.yaml"

# ---------------------------------------------------------------------------
# Startup script template (Compute Engine backend)
# Runs inside the VM as root; self-deletes on exit (success or failure).
# ---------------------------------------------------------------------------
_CE_STARTUP_SCRIPT = """\
#!/bin/bash
exec >> /var/log/startup-script.log 2>&1

INSTANCE_NAME="{instance_name}"
ZONE_META=$(curl -sf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/zone" | awk -F/ '{{print $NF}}')

cleanup() {{
    echo "[automend] $(date) -- self-deleting ${{INSTANCE_NAME}} in ${{ZONE_META}}"
    gcloud compute instances delete "${{INSTANCE_NAME}}" --zone="${{ZONE_META}}" --quiet || true
}}
trap cleanup EXIT
set -e

echo "[automend] $(date) -- Sweep trial {trial_index} starting"
echo "[automend] Sweep ID : {sweep_id}"
echo "[automend] Image    : {image}"

# Wait for Docker daemon (Deep Learning VMs finish setup before script runs,
# but adding a guard is safer)
until docker info >/dev/null 2>&1; do
    echo "[automend] Waiting for Docker daemon..."
    sleep 5
done
echo "[automend] Docker ready"

# Authenticate Docker against Artifact Registry
gcloud auth configure-docker {region}-docker.pkg.dev --quiet
echo "[automend] Docker auth configured"

# Pull image
docker pull "{image}"
echo "[automend] Image pulled"

# Run the W&B sweep agent (--count 1 means one trial per VM)
docker run --rm --gpus all \
    -e WANDB_PROJECT="{wandb_project}" \
    -e WANDB_ENTITY="{wandb_entity}" \
    -e WANDB_START_METHOD="thread" \
    -e PYTHONUNBUFFERED="1" \
    -e PYTHONPATH="/workspace" \
    -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    "{image}" \
    python /workspace/gcp/scripts/run_agent.py "{sweep_id}" "{count}"

echo "[automend] $(date) -- All {count} trial(s) complete"
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image(tag: str) -> str:
    base = IMAGE_URI.rsplit(":", 1)[0]
    return f"{base}:{tag}"


def _build_startup_script(
    sweep_id: str,
    image: str,
    instance_name: str,
    count: int,
) -> str:
    return _CE_STARTUP_SCRIPT.format(
        instance_name=instance_name,
        trial_index=1,
        sweep_id=sweep_id,
        image=image,
        region=REGION,
        wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,
        count=count,
    )


# ---------------------------------------------------------------------------
# W&B sweep creation
# ---------------------------------------------------------------------------

def create_sweep() -> str:
    """
    Auto-create a W&B sweep from the sweep config YAML.

    Returns:
        Full sweep path: entity/project/sweep_id
    """
    try:
        import wandb
    except ImportError:
        print("wandb not installed. Run: pip install wandb")
        sys.exit(1)

    import yaml
    with open(_SWEEP_CONFIG) as f:
        sweep_cfg = yaml.safe_load(f)

    print(f"Creating W&B sweep from: {_SWEEP_CONFIG}")
    sweep_id = wandb.sweep(
        sweep=sweep_cfg,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
    )
    full_sweep_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}"
    print(f"Sweep created: {full_sweep_path}")
    print(f"W&B URL: https://wandb.ai/{full_sweep_path}")
    return full_sweep_path


# ---------------------------------------------------------------------------
# Vertex AI backend
# ---------------------------------------------------------------------------

def submit_sweep_trial(
    sweep_id: str,
    trial_index: int,
    image_tag: str,
    dry_run: bool = False,
):
    """Submit a single W&B sweep trial as a Vertex AI Custom Training Job.

    Returns the job object (for log streaming) or None for dry-run.
    """
    try:
        from google.cloud import aiplatform
    except ImportError:
        print("google-cloud-aiplatform not installed. Run: pip install google-cloud-aiplatform")
        sys.exit(1)

    image = _image(image_tag)
    job_display_name = f"automend-sweep-trial-{trial_index:02d}"

    command = [
        "python", "/workspace/gcp/scripts/run_agent.py", sweep_id
    ]

    worker_pool_spec = {
        "machine_spec": {
            "machine_type":      TRAIN_MACHINE_TYPE,
            "accelerator_type":  TRAIN_ACCELERATOR,
            "accelerator_count": TRAIN_ACCEL_COUNT,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": image,
            "command": command,
            "env": [
                {"name": "WANDB_PROJECT",               "value": WANDB_PROJECT},
                {"name": "WANDB_ENTITY",                "value": WANDB_ENTITY},
                {"name": "WANDB_START_METHOD",          "value": "thread"},
                {"name": "PYTHONUNBUFFERED",            "value": "1"},
                {"name": "PYTHONPATH",                  "value": "/workspace"},
                {"name": "PYTORCH_CUDA_ALLOC_CONF",     "value": "expandable_segments:True"},
            ],
        },
    }

    if dry_run:
        print(f"  [DRY-RUN] Would submit Vertex AI job: {job_display_name}")
        print(f"    Image   : {image}")
        print(f"    Machine : {TRAIN_MACHINE_TYPE} + {TRAIN_ACCELERATOR} x{TRAIN_ACCEL_COUNT}")
        print(f"    Command : wandb agent --count 1 {sweep_id}")
        return None

    aiplatform.init(project=PROJECT_ID, location=REGION)

    job = aiplatform.CustomJob(
        display_name=job_display_name,
        worker_pool_specs=[worker_pool_spec],
        staging_bucket="gs://automend-model2/staging",
    )

    job.submit(service_account=TRAINER_SA)
    print(f"  Submitted: {job_display_name} — resource: {job.resource_name}")
    return job


def _stream_vertex_logs(job, trial_index: int) -> None:
    """Block until a Vertex AI job finishes, streaming its Cloud Logging output."""
    try:
        job.stream_logs()
    except KeyboardInterrupt:
        print(f"\n[vertex-t{trial_index:02d}] Detached from logs. Job is still running on Vertex AI.", flush=True)
    except Exception as exc:
        print(f"[vertex-t{trial_index:02d}] log stream ended: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Compute Engine backend
# ---------------------------------------------------------------------------

def submit_sweep_trial_ce(
    sweep_id: str,
    count: int,
    image_tag: str,
    zone: str,
    dry_run: bool = False,
):
    """Submit a single Compute Engine VM that runs `count` sequential W&B trials.

    The VM:
      - Uses a Deep Learning image (CUDA 12.8 + Docker + NVIDIA toolkit)
      - Pulls the training image from Artifact Registry
      - Runs: wandb agent --count {count} <sweep_id>  (sequential trials, one GPU)
      - Self-deletes when all trials finish (success or failure)

    Returns the instance name (str) for log streaming, or None for dry-run.
    """
    try:
        from google.cloud import compute_v1
    except ImportError:
        print("google-cloud-compute not installed. Run: pip install google-cloud-compute")
        sys.exit(1)

    image = _image(image_tag)
    instance_name = f"automend-sweep-{int(time.time())}"
    startup_script = _build_startup_script(sweep_id, image, instance_name, count)

    # A2 (A100) and G2 (L4) machine families have the GPU built into the machine type.
    # N1 machines (n1-standard-*) need a separate accelerator attachment for T4/P4/V100.
    _gpu_builtin = CE_MACHINE_TYPE.startswith("a2-") or CE_MACHINE_TYPE.startswith("g2-")
    _gpu_label   = (
        "A100 40GB (built-in)" if CE_MACHINE_TYPE.startswith("a2-") else
        "L4 24GB (built-in)"   if CE_MACHINE_TYPE.startswith("g2-") else
        f"nvidia-tesla-t4 x{TRAIN_ACCEL_COUNT}"
    )

    if dry_run:
        gpu_label = _gpu_label
        print(f"  [DRY-RUN] Would create CE instance: {instance_name}")
        print(f"    Zone    : {zone}")
        print(f"    Machine : {CE_MACHINE_TYPE}  ({_gpu_label})")
        print(f"    Trials  : {count} sequential (wandb agent --count {count})")
        print(f"    Image   : {image}")
        print(f"    Disk    : {CE_DISK_IMAGE_PROJECT}/{CE_DISK_IMAGE_FAMILY} ({CE_DISK_SIZE_GB} GB)")
        return None

    instances_client = compute_v1.InstancesClient()

    # --- boot disk (Deep Learning VM image: CUDA 12.4 + Docker + NVIDIA toolkit) ---
    disk = compute_v1.AttachedDisk()
    disk.auto_delete = True
    disk.boot = True
    init_params = compute_v1.AttachedDiskInitializeParams()
    init_params.disk_size_gb = CE_DISK_SIZE_GB
    init_params.source_image = (
        f"projects/{CE_DISK_IMAGE_PROJECT}/global/images/family/{CE_DISK_IMAGE_FAMILY}"
    )
    disk.initialize_params = init_params

    # --- network interface (external IP needed for Docker pull from Artifact Registry) ---
    nic = compute_v1.NetworkInterface()
    nic.name = "global/networks/default"
    access = compute_v1.AccessConfig()
    access.name = "External NAT"
    access.type_ = "ONE_TO_ONE_NAT"
    nic.access_configs = [access]

    # --- scheduling (all GPU instances require TERMINATE on maintenance, no auto-restart) ---
    sched = compute_v1.Scheduling()
    sched.on_host_maintenance = "TERMINATE"
    sched.automatic_restart = False

    # --- startup script metadata ---
    metadata = compute_v1.Metadata()
    item = compute_v1.Items()
    item.key = "startup-script"
    item.value = startup_script
    metadata.items = [item]

    # --- service account ---
    sa = compute_v1.ServiceAccount()
    sa.email = TRAINER_SA
    sa.scopes = ["https://www.googleapis.com/auth/cloud-platform"]

    # --- assemble instance ---
    instance = compute_v1.Instance()
    instance.name = instance_name
    instance.machine_type = f"zones/{zone}/machineTypes/{CE_MACHINE_TYPE}"
    instance.disks = [disk]
    instance.network_interfaces = [nic]
    instance.scheduling = sched
    instance.metadata = metadata
    instance.service_accounts = [sa]

    # A2 (A100) and G2 (L4): GPU built into machine type — no AcceleratorConfig needed.
    # N1 machines (T4/P4/V100): GPU must be attached separately.
    if not _gpu_builtin:
        accel = compute_v1.AcceleratorConfig()
        accel.accelerator_count = TRAIN_ACCEL_COUNT
        accel.accelerator_type = f"zones/{zone}/acceleratorTypes/nvidia-tesla-t4"
        instance.accelerators = [accel]

    # Submit (operation.result() waits for the insert op, not for the job to finish)
    operation = instances_client.insert(
        project=PROJECT_ID,
        zone=zone,
        instance_resource=instance,
    )
    operation.result()

    console_url = (
        f"https://console.cloud.google.com/compute/instancesDetail"
        f"/zones/{zone}/instances/{instance_name}?project={PROJECT_ID}"
    )
    print(f"  Created CE instance: {instance_name}")
    print(f"  Console: {console_url}")
    return instance_name


def _print_crash_warning(instance_name: str, zone: str, trial_index: int) -> None:
    """Print a loud warning when a CE instance terminates without self-deleting (crash)."""
    border = "=" * 70
    lines = [
        "",
        border,
        f"  !! CRASH DETECTED — trial {trial_index:02d} — instance did NOT self-delete !!",
        border,
        f"  Instance : {instance_name}",
        f"  Zone     : {zone}",
        f"  Status   : TERMINATED (instance still exists on GCE)",
        f"  Likely   : kernel panic, OS-level OOM kill, or GPU driver crash",
        f"  Trial    : MAY BE INCOMPLETE — check W&B for a result",
        "",
        f"  View full serial port log:",
        f"    gcloud compute instances get-serial-port-output {instance_name} \\",
        f"      --zone={zone} --project={PROJECT_ID} --port=1",
        "",
        f"  Delete the dead instance to stop billing:",
        f"    gcloud compute instances delete {instance_name} \\",
        f"      --zone={zone} --project={PROJECT_ID}",
        border,
        "",
    ]
    print("\n".join(lines), flush=True)


def _stream_ce_logs(instance_name: str, zone: str, trial_index: int) -> None:
    """Poll serial port output for a CE instance until it self-deletes or crashes."""
    prefix = f"[ce-t{trial_index:02d}]"
    reconnect_cmd = (
        f"gcloud compute instances get-serial-port-output {instance_name}"
        f" --zone={zone} --project={PROJECT_ID} --port=1"
    )

    # Print reconnect info immediately — if the terminal is killed, the VM still runs
    # and these commands let you re-attach to the output later.
    print(f"{prefix} Streaming serial port output (15 s poll).", flush=True)
    print(f"{prefix} Ctrl+C detaches — VM keeps running. Reconnect with:", flush=True)
    print(f"{prefix}   {reconnect_cmd}", flush=True)
    print(flush=True)

    printed_lines = 0
    poll_interval = 15

    while True:
        # Fetch full serial port output (accumulates from boot)
        result = subprocess.run(
            [_GCLOUD, "compute", "instances", "get-serial-port-output",
             instance_name, f"--zone={zone}", f"--project={PROJECT_ID}", "--port=1"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            all_lines = result.stdout.splitlines()
            for line in all_lines[printed_lines:]:
                print(f"{prefix} {line}", flush=True)
            printed_lines = len(all_lines)

        # Check instance status
        status_result = subprocess.run(
            [_GCLOUD, "compute", "instances", "describe",
             instance_name, f"--zone={zone}", f"--project={PROJECT_ID}", "--format=value(status)"],
            capture_output=True, text=True,
        )

        if status_result.returncode != 0:
            # describe failed → instance is gone → self-deleted successfully
            print(f"{prefix} ✓ Instance self-deleted — trial complete.", flush=True)
            break

        status = status_result.stdout.strip()

        if status == "STOPPING":
            # Deletion in progress — poll faster, don't sleep long
            time.sleep(5)
            continue

        if status in ("TERMINATED", "STOPPED"):
            # Instance stopped. Flush final logs then decide: clean exit or crash?
            time.sleep(5)
            final = subprocess.run(
                [_GCLOUD, "compute", "instances", "get-serial-port-output",
                 instance_name, f"--zone={zone}", f"--project={PROJECT_ID}", "--port=1"],
                capture_output=True, text=True,
            )
            if final.returncode == 0:
                all_lines = final.stdout.splitlines()
                for line in all_lines[printed_lines:]:
                    print(f"{prefix} {line}", flush=True)
                printed_lines = len(all_lines)

            # Wait for the self-delete gcloud command inside the VM to complete.
            # If the trap fired, the instance will disappear within ~15 s.
            time.sleep(15)
            still_there = subprocess.run(
                [_GCLOUD, "compute", "instances", "describe",
                 instance_name, f"--zone={zone}", f"--project={PROJECT_ID}", "--format=value(status)"],
                capture_output=True, text=True,
            )
            if still_there.returncode != 0:
                # Gone — self-delete completed after the container exited
                print(f"{prefix} ✓ Instance self-deleted — trial complete.", flush=True)
            else:
                # Still there — trap did NOT fire — VM crashed hard (kernel panic / OOM)
                _print_crash_warning(instance_name, zone, trial_index)
            break

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-create W&B sweep and launch trials on Vertex AI or Compute Engine."
    )
    parser.add_argument(
        "--backend",
        choices=["vertex", "compute-engine"],
        default="vertex",
        help=(
            "Backend for running sweep trials. "
            "'vertex' (default) uses Vertex AI Custom Training Jobs. "
            "'compute-engine' provisions GPU VMs directly and self-deletes them when done."
        ),
    )
    parser.add_argument(
        "--sweep-id",
        default=None,
        help=(
            "Resume an existing sweep (entity/project/sweep_id). "
            "If omitted, a new sweep is created automatically."
        ),
    )
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Number of parallel sweep trials to launch (default: 10)",
    )
    parser.add_argument(
        "--image-tag", default="latest",
        help="Docker image tag (default: latest)",
    )
    parser.add_argument(
        "--zone", default=CE_ZONE,
        help=(
            f"Compute Engine zone (only used with --backend compute-engine, "
            f"default: {CE_ZONE})"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print job/instance configs but do NOT submit to GCP",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Auto-create sweep if no sweep-id provided
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Resuming existing sweep: {sweep_id}")
    else:
        sweep_id = create_sweep()

    print()
    backend_label = "Vertex AI" if args.backend == "vertex" else f"Compute Engine ({args.zone})"
    print(f"{'[DRY-RUN] ' if args.dry_run else ''}Launching {args.trials} sweep trials on {backend_label}")
    print(f"  Sweep ID : {sweep_id}")
    print(f"  Image    : {_image(args.image_tag)}")
    if args.backend == "vertex":
        print(f"  Machine  : {TRAIN_MACHINE_TYPE} + {TRAIN_ACCELERATOR} x{TRAIN_ACCEL_COUNT}")
    else:
        _ce_gpu_label = (
            "A100 40GB (built-in)" if CE_MACHINE_TYPE.startswith("a2-") else
            "L4 24GB (built-in)"   if CE_MACHINE_TYPE.startswith("g2-") else
            f"nvidia-tesla-t4 x{TRAIN_ACCEL_COUNT}"
        )
        print(f"  Machine  : {CE_MACHINE_TYPE}  ({_ce_gpu_label})")
    print(f"  SA       : {TRAINER_SA}")
    print()

    vertex_jobs: list[tuple[int, object]] = []
    ce_instances: list[tuple[int, str]] = []

    if args.backend == "vertex":
        # Vertex AI: N parallel jobs, each runs 1 trial
        for i in range(1, args.trials + 1):
            job = submit_sweep_trial(
                sweep_id=sweep_id,
                trial_index=i,
                image_tag=args.image_tag,
                dry_run=args.dry_run,
            )
            if job is not None:
                vertex_jobs.append((i, job))
    else:
        # Compute Engine: 1 VM, N sequential trials (1 GPU, no parallel quota needed)
        print(f"  Mode     : 1 VM × {args.trials} sequential trials (wandb agent --count {args.trials})")
        instance_name = submit_sweep_trial_ce(
            sweep_id=sweep_id,
            count=args.trials,
            image_tag=args.image_tag,
            zone=args.zone,
            dry_run=args.dry_run,
        )
        if instance_name is not None:
            ce_instances.append((1, instance_name))

    print()
    if args.dry_run:
        print("Dry-run complete — no jobs submitted.")
        return

    print(f"All {args.trials} trials submitted.")

    # --- Print a persistent summary BEFORE streaming starts.
    # If the terminal is killed mid-stream, the user still has these commands
    # in their scrollback to reconnect or check status. ---
    print()
    if vertex_jobs:
        print("Vertex AI monitor (jobs run independently of this terminal):")
        print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")

    if ce_instances:
        print("Compute Engine instances (VMs run independently of this terminal):")
        print(f"  https://console.cloud.google.com/compute/instances?project={PROJECT_ID}")
        print()
        print("  Reconnect to logs at any time:")
        for _, name in ce_instances:
            print(
                f"    gcloud compute instances get-serial-port-output"
                f" {name} --zone={args.zone} --project={PROJECT_ID} --port=1"
            )

    print()
    print("Streaming logs below. Ctrl+C detaches — all jobs keep running.\n")

    # --- Vertex AI: stream logs in parallel threads ---
    if vertex_jobs:
        threads = [
            threading.Thread(target=_stream_vertex_logs, args=(job, i), daemon=True)
            for i, job in vertex_jobs
        ]
        for t in threads:
            t.start()
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print(
                "\nDetached from Vertex AI log stream."
                " Jobs are still running — monitor at:"
                f"\n  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}"
            )

    # --- Compute Engine: stream serial port output in parallel threads ---
    if ce_instances:
        ce_threads = [
            threading.Thread(target=_stream_ce_logs, args=(name, args.zone, i), daemon=True)
            for i, name in ce_instances
        ]
        for t in ce_threads:
            t.start()
        try:
            for t in ce_threads:
                t.join()
        except KeyboardInterrupt:
            print("\nDetached from CE log stream. VM is still running. Reconnect with:")
            for _, name in ce_instances:
                print(
                    f"  gcloud compute instances get-serial-port-output"
                    f" {name} --zone={args.zone} --project={PROJECT_ID} --port=1"
                )

    print()
    print("When all trials complete, fetch best config + train:")
    print(f"  python model_2_training/scripts/fetch_best_config.py --sweep-id {sweep_id}")


if __name__ == "__main__":
    main()
