"""
Ray and scalability configuration for the Automend MLOps pipeline.

Provides centralized settings for Ray initialization, per-dataset
chunk/sample sizes, and environment detection (local vs cloud).

Usage:
    from src.config.ray_config import init_ray, get_dataset_config
    
    init_ray()
    cfg = get_dataset_config("ds1")
    sample_size = cfg["sample_size"]
"""

import os
import logging
from typing import Optional

logger = logging.getLogger("automend.ray_config")

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def _detect_environment() -> str:
    if os.environ.get("RAY_ADDRESS"):
        return "cluster"
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return "kubernetes"
    return "local"


ENVIRONMENT = _detect_environment()

# ---------------------------------------------------------------------------
# Ray init defaults (overridable via env vars)
# ---------------------------------------------------------------------------

RAY_CONFIG = {
    "local": {
        "num_cpus": int(os.environ.get("RAY_NUM_CPUS", 4)),
        "object_store_memory": int(os.environ.get("RAY_OBJECT_STORE_MB", 512)) * 1024 * 1024,
        "dashboard_host": "0.0.0.0",
        "include_dashboard": False,
    },
    "cluster": {
        "address": os.environ.get("RAY_ADDRESS", "auto"),
    },
    "kubernetes": {
        "address": os.environ.get("RAY_ADDRESS", "auto"),
    },
}

# ---------------------------------------------------------------------------
# Per-dataset scalability knobs
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "ds1": {
        "sample_size": int(os.environ.get("DS1_SAMPLE_SIZE", 100)),
        "window_size": int(os.environ.get("DS1_WINDOW_SIZE", 5)),
    },
    "ds2": {
        "sample_size": int(os.environ.get("DS2_SAMPLE_SIZE", 2000)),
    },
    "ds3": {
        "chunk_size": int(os.environ.get("DS3_CHUNK_SIZE", 100_000)),
        "sample_size": int(os.environ.get("DS3_SAMPLE_SIZE", 0)),  # 0 = all
    },
    "ds4": {
        "num_workers": int(os.environ.get("DS4_NUM_WORKERS", 4)),
        "max_concurrent_per_worker": int(os.environ.get("DS4_MAX_CONCURRENT", 5)),
    },
    "ds5": {
        "sample_size": int(os.environ.get("DS5_SAMPLE_SIZE", 5000)),
        "random_seed": 42,
    },
    "ds6": {
        "sample_size": int(os.environ.get("DS6_SAMPLE_SIZE", 20_000)),
        "chunk_size": int(os.environ.get("DS6_CHUNK_SIZE", 1000)),
    },
}


def get_dataset_config(dataset_key: str) -> dict:
    return DATASET_CONFIGS.get(dataset_key, {})


def init_ray(override: Optional[dict] = None, ignore_reinit: bool = True) -> None:
    """Initialize Ray with environment-appropriate settings."""
    import ray

    if ray.is_initialized():
        if ignore_reinit:
            logger.info("Ray already initialized — skipping")
            return
        ray.shutdown()

    env = _detect_environment()
    kwargs = dict(RAY_CONFIG.get(env, {}))
    if override:
        kwargs.update(override)

    logger.info("Initializing Ray (env=%s): %s", env, kwargs)
    ray.init(**kwargs, ignore_reinit_error=ignore_reinit)
    logger.info("Ray initialized — %s", ray.cluster_resources())


def shutdown_ray() -> None:
    import ray
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray shut down")
