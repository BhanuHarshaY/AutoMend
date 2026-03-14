"""Pytest configuration for DS2 Loghub tests."""
import sys
from pathlib import Path

import pytest

# Must be done before any DS2 imports
DS2_ROOT = Path(__file__).resolve().parent.parent
DS2_SRC = DS2_ROOT / "src"

# Add DS2_ROOT first so that imports like "from src.utils..." work within the dataset's modules
if str(DS2_ROOT) not in sys.path:
    sys.path.insert(0, str(DS2_ROOT))
# Add DS2's src directory to path for direct imports like "from utils..."
if str(DS2_SRC) not in sys.path:
    sys.path.insert(0, str(DS2_SRC))


@pytest.fixture(autouse=True, scope="module")
def _ray_init():
    import ray
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
