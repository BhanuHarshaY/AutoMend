"""
Shared data acquisition orchestrator.

Each DAG calls ``ensure_data(dataset_key, raw_dir, project_root)`` during its
acquire step.  The function:

1. Checks if data already exists locally.
2. If not, tries ``dvc pull``.
3. If still missing, dispatches to the right seed or download function
   based on ``PIPELINE_DATA_MODE`` (read via ``src.config.data_mode``).
4. Versions the newly acquired data with DVC.

Returns a dict with ``{"status": "cached"|"seeded"|"downloaded"|"error", ...}``.
"""

import importlib
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

_MODE_MARKER = ".data_mode"


def _project_root_from(raw_dir: Path) -> Path:
    """Derive project root: raw_dir is typically data/raw/ds{N}_name."""
    return raw_dir.parent.parent.parent


def _has_data(raw_dir: Path) -> bool:
    return raw_dir.exists() and any(raw_dir.iterdir())


def _read_mode_marker(raw_dir: Path) -> Optional[str]:
    marker = raw_dir / _MODE_MARKER
    if marker.exists():
        return marker.read_text().strip()
    return None


def _write_mode_marker(raw_dir: Path, mode: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / _MODE_MARKER).write_text(mode + "\n")


def ensure_data(
    dataset_key: str,
    raw_dir: Union[str, Path],
    project_root: Union[str, Path, None] = None,
) -> dict:
    """
    Main entry point.  Ensures raw data is present for *dataset_key*.

    Parameters
    ----------
    dataset_key : str
        One of "ds1" .. "ds6".
    raw_dir : Path
        Absolute path to the dataset's raw data directory.
    project_root : Path | None
        Project root for DVC commands.  Derived from *raw_dir* if omitted.
    """
    raw_dir = Path(raw_dir)
    if project_root is None:
        project_root = _project_root_from(raw_dir)
    project_root = Path(project_root)

    # Ensure project root on sys.path for imports
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.config.data_mode import get_data_mode, get_acquire_config
    from src.utils.dvc_utils import check_raw_data_exists, version_raw_data

    mode = get_data_mode()
    cfg = get_acquire_config(dataset_key)

    # 1. Already present?
    if check_raw_data_exists(raw_dir, project_root=project_root):
        prev_mode = _read_mode_marker(raw_dir)
        if prev_mode == mode:
            logger.info("[%s] Raw data found (mode=%s). Skipping acquire.", dataset_key, mode)
            return {"status": "cached", "mode": mode}
        logger.warning(
            "[%s] Mode changed (%s -> %s). Clearing stale data for re-acquisition.",
            dataset_key, prev_mode, mode,
        )
        shutil.rmtree(raw_dir)

    # 2. Acquire based on mode
    logger.info("[%s] Raw data missing — acquiring in '%s' mode.", dataset_key, mode)
    raw_dir.mkdir(parents=True, exist_ok=True)

    dispatcher = _DISPATCH.get(dataset_key)
    if dispatcher is None:
        raise ValueError(f"Unknown dataset_key: {dataset_key}")

    result = dispatcher(raw_dir, project_root, mode, cfg)

    # 3. Record which mode produced this data
    if result.get("status") not in ("error",):
        _write_mode_marker(raw_dir, mode)

    # 4. Version with DVC after successful acquisition
    if result.get("status") not in ("error",):
        try:
            version_raw_data(str(raw_dir), cwd=str(project_root))
            logger.info("[%s] Raw data versioned with DVC.", dataset_key)
        except Exception as exc:
            logger.warning("[%s] DVC versioning failed: %s", dataset_key, exc)

    return result


# ---------------------------------------------------------------------------
# Per-dataset dispatchers
# ---------------------------------------------------------------------------

def _acquire_ds1(raw_dir: Path, project_root: Path, mode: str, cfg: dict) -> dict:
    if mode == "full":
        raise FileNotFoundError(
            f"DATA_MODE=full for DS1 (Alibaba): please download the Alibaba "
            f"Cluster Trace 2017 dataset manually and place CSV files in {raw_dir}/"
        )
    num_rows = cfg.get("num_rows", 100)
    scripts = project_root / "src" / "dataset_1_alibaba" / "scripts"
    sys.path.insert(0, str(scripts))
    from seed_data import generate_server_usage, generate_batch_task, generate_server_event

    for name, gen_fn in [
        ("server_usage_sample.csv", generate_server_usage),
        ("batch_task_sample.csv", generate_batch_task),
        ("server_event_sample.csv", generate_server_event),
    ]:
        rows = gen_fn(num_rows)
        (raw_dir / name).write_text("\n".join(rows) + "\n")

    return {"status": "seeded", "mode": mode, "num_rows": num_rows}


def _acquire_ds2(raw_dir: Path, project_root: Path, mode: str, cfg: dict) -> dict:
    source = cfg.get("source", "seed")
    if source == "seed":
        ds2_ingest = project_root / "src" / "dataset_2_loghub" / "src" / "ingest"
        sys.path.insert(0, str(ds2_ingest))
        from seed_data import generate_all
        generate_all(raw_dir / "loghub", num_rows=cfg.get("num_rows", 50))
        return {"status": "seeded", "mode": mode}
    else:
        ds2_src = project_root / "src" / "dataset_2_loghub" / "src"
        sys.path.insert(0, str(ds2_src))
        ds2_ingest = ds2_src / "ingest"
        sys.path.insert(0, str(ds2_ingest))
        from download_data import download_all
        ok = download_all()
        return {"status": "downloaded" if ok else "error", "mode": mode}


def _acquire_ds3(raw_dir: Path, project_root: Path, mode: str, cfg: dict) -> dict:
    scripts = project_root / "src" / "dataset_3_stackoverflow" / "scripts"
    sys.path.insert(0, str(scripts))
    source = cfg.get("source", "seed")

    if source == "seed":
        from seed_data import generate_questions, generate_answers, EXTERNAL_DIR
        import csv
        EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        num = cfg.get("num_rows", 50)
        questions = generate_questions(num)
        q_file = EXTERNAL_DIR / "Stack_Qns_pl.csv"
        with open(q_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["Id", "Title", "Body", "Tags", "Score", "ViewCount", "AcceptedAnswerId", "CreationDate"])
            w.writeheader()
            w.writerows(questions)
        answers = generate_answers(questions)
        a_file = EXTERNAL_DIR / "Stack_Ans_pl.csv"
        with open(a_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["AnswerId", "QuestionId", "AnswerBody", "AnswerScore"])
            w.writeheader()
            w.writerows(answers)
        from data_acquisition import run_acquisition
        stats = run_acquisition(use_csv=True)
        return {"status": "seeded", "mode": mode, **stats}
    else:
        from data_acquisition import run_acquisition
        max_q = cfg.get("max_questions", 0)
        stats = run_acquisition(use_csv=False, max_questions=max_q)
        return {"status": "downloaded", "mode": mode, **stats}


def _acquire_ds4(raw_dir: Path, project_root: Path, mode: str, cfg: dict) -> dict:
    ds4_root = project_root / "src" / "dataset_4_synthetic"
    sys.path.insert(0, str(ds4_root))
    scripts = ds4_root / "scripts"
    sys.path.insert(0, str(scripts))

    from seed_prompts import seed_prompts
    count = seed_prompts(
        prompt_set=cfg.get("prompt_set", "default"),
        prompt_count=cfg.get("prompt_count", 15),
    )
    return {"status": "seeded", "mode": mode, "prompts_seeded": count}


def _acquire_ds5(raw_dir: Path, project_root: Path, mode: str, cfg: dict) -> dict:
    source = cfg.get("source", "seed")
    scripts = project_root / "src" / "dataset_5_glaive" / "scripts"
    sys.path.insert(0, str(scripts))

    if source == "seed":
        from seed_data import generate_all
        output = raw_dir / "glaive_raw.jsonl"
        count = generate_all(output, num_records=cfg.get("num_records", 100))
        return {"status": "seeded", "mode": mode, "record_count": count}
    else:
        from data_acquisition import fetch_and_save
        sample_size = cfg.get("sample_size", 5000)
        output = raw_dir / "glaive_raw.jsonl"
        count = fetch_and_save(sample_size=sample_size, output_file=output)
        return {"status": "downloaded", "mode": mode, "record_count": count}


def _acquire_ds6(raw_dir: Path, project_root: Path, mode: str, cfg: dict) -> dict:
    source = cfg.get("source", "seed")

    if source == "seed":
        seed_dir = project_root / "src" / "dataset_6_the_stack" / "scripts" / "download"
        sys.path.insert(0, str(seed_dir))
        from seed_data import generate_all
        count = generate_all(raw_dir, num_records=cfg.get("num_records", 50))
        return {"status": "seeded", "mode": mode, "record_count": count}
    else:
        ds6_root = project_root / "src" / "dataset_6_the_stack"
        sys.path.insert(0, str(ds6_root))
        from scripts.download.stack_iac_sample import download
        download(sample_size=cfg.get("sample_size", 20000))
        return {"status": "downloaded", "mode": mode}


_DISPATCH = {
    "ds1": _acquire_ds1,
    "ds2": _acquire_ds2,
    "ds3": _acquire_ds3,
    "ds4": _acquire_ds4,
    "ds5": _acquire_ds5,
    "ds6": _acquire_ds6,
}
