"""
Seed script for DS1 (Alibaba Cluster Trace 2017) - generates sample CSVs for E2E testing.

Creates 3 sample CSV files with realistic data matching the Alibaba Cluster Trace schema:
- server_usage_sample.csv (100 rows): time series of server resource utilization
- batch_task_sample.csv (100 rows): batch job task status records
- server_event_sample.csv (100 rows): server lifecycle events

Usage:
    python -m src.dataset_1_alibaba.scripts.seed_data
    python src/dataset_1_alibaba/scripts/seed_data.py
"""

import sys
from pathlib import Path
import random
import numpy as np

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config.paths import get_ds1_raw_dir
    RAW_DIR = get_ds1_raw_dir()
except ImportError:
    RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ds1_alibaba"

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Output files
SERVER_USAGE_FILE = RAW_DIR / "server_usage_sample.csv"
BATCH_TASK_FILE = RAW_DIR / "batch_task_sample.csv"
SERVER_EVENT_FILE = RAW_DIR / "server_event_sample.csv"

# Constants for realistic data generation
NUM_ROWS = 100
NUM_MACHINES = 20
STATUSES = ["Terminated", "Failed", "Waiting", "Running", "Unknown"]
STATUS_WEIGHTS = [0.3, 0.1, 0.1, 0.45, 0.05]
EVENT_TYPES = ["add", "remove", "failure"]
EVENT_WEIGHTS = [0.5, 0.35, 0.15]


def generate_server_usage(num_rows: int = NUM_ROWS) -> list[str]:
    """
    Generate server_usage_sample.csv with columns:
    time_stamp, machine_id, cpu_util_percent, mem_util_percent, net_in, net_out, disk_io_percent, extra
    """
    rows = []
    base_time = 0  # Relative timestamp in seconds
    
    for i in range(num_rows):
        time_stamp = base_time + i * 300  # 5-minute intervals
        machine_id = f"m_{random.randint(1, NUM_MACHINES):05d}"
        
        # Generate realistic utilization patterns
        # Some machines have high usage (potential anomalies)
        if random.random() < 0.15:  # 15% high usage
            cpu_util = round(random.uniform(70, 100), 2)
            mem_util = round(random.uniform(70, 100), 2)
        elif random.random() < 0.1:  # 10% low usage
            cpu_util = round(random.uniform(0, 30), 2)
            mem_util = round(random.uniform(0, 30), 2)
        else:  # Normal usage
            cpu_util = round(random.uniform(20, 70), 2)
            mem_util = round(random.uniform(30, 70), 2)
        
        net_in = round(random.uniform(0, 1000), 2)
        net_out = round(random.uniform(0, 1000), 2)
        disk_io = round(random.uniform(0, 100), 2)
        extra = ""
        
        row = f"{time_stamp},{machine_id},{cpu_util},{mem_util},{net_in},{net_out},{disk_io},{extra}"
        rows.append(row)
    
    return rows


def generate_batch_task(num_rows: int = NUM_ROWS) -> list[str]:
    """
    Generate batch_task_sample.csv with columns:
    start_time, end_time, inst_num, task_type, job_id, status, plan_cpu, plan_mem
    """
    rows = []
    base_time = 0
    
    for i in range(num_rows):
        start_time = base_time + random.randint(0, 86400)  # Random time within a day
        duration = random.randint(60, 3600)  # 1 minute to 1 hour
        end_time = start_time + duration
        
        inst_num = random.randint(1, 10)
        task_type = random.randint(1, 12)  # 12 task types in Alibaba trace
        job_id = f"j_{random.randint(1, 1000):06d}"
        
        # Status with weighted probability
        status = random.choices(STATUSES, weights=STATUS_WEIGHTS)[0]
        
        # Plan CPU (0-100 scale)
        plan_cpu = round(random.uniform(10, 100), 2)
        # Plan memory (normalized 0-1)
        plan_mem = round(random.uniform(0.1, 0.9), 4)
        
        row = f"{start_time},{end_time},{inst_num},{task_type},{job_id},{status},{plan_cpu},{plan_mem}"
        rows.append(row)
    
    return rows


def generate_server_event(num_rows: int = NUM_ROWS) -> list[str]:
    """
    Generate server_event_sample.csv with columns:
    time_stamp, machine_id, event_type, event_detail, plan_cpu, plan_mem, extra
    """
    rows = []
    base_time = 0
    
    for i in range(num_rows):
        time_stamp = base_time + i * 600  # 10-minute intervals
        machine_id = f"m_{random.randint(1, NUM_MACHINES):05d}"
        
        # Event type with weighted probability
        event_type = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS)[0]
        
        # Event detail based on type
        if event_type == "add":
            event_detail = "server_added"
        elif event_type == "remove":
            event_detail = "server_removed"
        else:  # failure
            event_detail = random.choice(["hardware_failure", "network_failure", "disk_failure"])
        
        plan_cpu = round(random.uniform(10, 100), 2)
        plan_mem = round(random.uniform(0.1, 0.9), 4)
        extra = ""
        
        row = f"{time_stamp},{machine_id},{event_type},{event_detail},{plan_cpu},{plan_mem},{extra}"
        rows.append(row)
    
    return rows


def main(num_rows: int | None = None):
    """Generate all seed data files for DS1.

    Parameters
    ----------
    num_rows : int | None
        Number of rows per CSV.  Defaults to ``NUM_ROWS`` (100).
    """
    n = num_rows if num_rows is not None else NUM_ROWS
    print(f"Generating DS1 (Alibaba) seed data ({n} rows)...")
    print(f"Output directory: {RAW_DIR}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    server_usage_rows = generate_server_usage(n)
    with open(SERVER_USAGE_FILE, "w") as f:
        f.write("\n".join(server_usage_rows) + "\n")
    print(f"  Created: {SERVER_USAGE_FILE.name} ({len(server_usage_rows)} rows)")

    batch_task_rows = generate_batch_task(n)
    with open(BATCH_TASK_FILE, "w") as f:
        f.write("\n".join(batch_task_rows) + "\n")
    print(f"  Created: {BATCH_TASK_FILE.name} ({len(batch_task_rows)} rows)")

    server_event_rows = generate_server_event(n)
    with open(SERVER_EVENT_FILE, "w") as f:
        f.write("\n".join(server_event_rows) + "\n")
    print(f"  Created: {SERVER_EVENT_FILE.name} ({len(server_event_rows)} rows)")

    print(f"\nDS1 seed data generation complete!")
    print(f"Files saved to: {RAW_DIR}")
    return True


if __name__ == "__main__":
    main()
