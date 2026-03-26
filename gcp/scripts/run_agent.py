"""
run_agent.py

Fetches WANDB_API_KEY from Secret Manager using the Python SDK
(no gcloud CLI required) then launches the W&B sweep agent.

Usage (inside container):
    python /workspace/gcp/scripts/run_agent.py <sweep_id>
"""
import os
import subprocess
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: run_agent.py <sweep_id> [count]")
        sys.exit(1)

    sweep_id   = sys.argv[1]
    count      = sys.argv[2] if len(sys.argv) > 2 else "1"
    project_id = os.environ.get("PROJECT_ID", "automend")

    # Fetch W&B API key from Secret Manager (uses Workload Identity / ADC automatically)
    from google.cloud import secretmanager
    client   = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(
        name=f"projects/{project_id}/secrets/WANDB_API_KEY/versions/latest"
    )
    os.environ["WANDB_API_KEY"] = response.payload.data.decode().strip()
    print(f"WANDB_API_KEY loaded from Secret Manager (project={project_id})")
    print(f"Running {count} sequential trial(s) for sweep: {sweep_id}")

    result = subprocess.call(["wandb", "agent", "--count", count, sweep_id])
    sys.exit(result)


if __name__ == "__main__":
    main()
