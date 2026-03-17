"""
Seed script for DS2 (LogHub) - generates synthetic log CSVs for dummy mode.

Creates structured CSV and templates CSV for each of the 5 systems
(Linux, HPC, HDFS, Hadoop, Spark) with realistic column schemas matching
the real LogHub 2K datasets.

Usage:
    python -m src.dataset_2_loghub.src.ingest.seed_data
"""

import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DS2_SRC = SCRIPT_DIR.parent
PROJECT_ROOT = DS2_SRC.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

random.seed(42)

# Column schemas per system (matching real LogHub structured CSVs)
SYSTEM_SCHEMAS = {
    "Linux": {
        "structured_cols": "LineId,Month,Date,Time,Level,Component,PID,Content,EventId,EventTemplate",
        "template_cols": "EventId,EventTemplate,Occurrences",
        "gen_structured": "_gen_linux_row",
        "gen_template": "_gen_linux_template",
    },
    "HPC": {
        "structured_cols": "LineId,LogId,Node,Component,State,Time,Flag,Content,EventId,EventTemplate",
        "template_cols": "EventId,EventTemplate,Occurrences",
        "gen_structured": "_gen_hpc_row",
        "gen_template": "_gen_hpc_template",
    },
    "HDFS": {
        "structured_cols": "LineId,Date,Time,Pid,Level,Component,Content,EventId,EventTemplate",
        "template_cols": "EventId,EventTemplate,Occurrences",
        "gen_structured": "_gen_hdfs_row",
        "gen_template": "_gen_hdfs_template",
    },
    "Hadoop": {
        "structured_cols": "LineId,Date,Time,Level,Process,Component,Content,EventId,EventTemplate",
        "template_cols": "EventId,EventTemplate,Occurrences",
        "gen_structured": "_gen_hadoop_row",
        "gen_template": "_gen_hadoop_template",
    },
    "Spark": {
        "structured_cols": "LineId,Date,Time,Level,Component,Content,EventId,EventTemplate",
        "template_cols": "EventId,EventTemplate,Occurrences",
        "gen_structured": "_gen_spark_row",
        "gen_template": "_gen_spark_template",
    },
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LEVELS = ["INFO", "WARN", "ERROR", "DEBUG", "FATAL"]
LEVEL_WEIGHTS = [0.5, 0.2, 0.15, 0.1, 0.05]

LINUX_COMPONENTS = ["sshd", "kernel", "CRON", "systemd", "sudo", "dhclient"]
LINUX_MESSAGES = [
    "Accepted publickey for user from 10.0.0.1 port 22 ssh2",
    "pam_unix(sshd:session): session opened for user root",
    "Failed password for invalid user admin from 10.0.0.2 port 22 ssh2",
    "Connection closed by 10.0.0.3 port 22 [preauth]",
    "session-1.scope: Succeeded.",
    "CMD (run-parts /etc/cron.hourly)",
]
LINUX_TEMPLATES = [
    "Accepted publickey for <*> from <*> port <*> ssh2",
    "pam_unix(sshd:session): session opened for user <*>",
    "Failed password for invalid user <*> from <*> port <*> ssh2",
    "Connection closed by <*> port <*> [preauth]",
    "<*>.scope: Succeeded.",
    "CMD (run-parts <*>)",
]

HPC_COMPONENTS = ["kernel", "ib", "ib_cm", "rtssp", "power"]
HPC_STATES = ["state_change.unavailable", "state_change.available", "normal", "error"]
HPC_MESSAGES = [
    "node-1234 unavailable - link down",
    "InfiniBand port active on node-5678",
    "Power supply warning on rack 3",
    "Memory ECC error corrected on node-9012",
    "Node rebooted successfully",
]
HPC_TEMPLATES = [
    "<*> unavailable - link down",
    "InfiniBand port active on <*>",
    "Power supply warning on rack <*>",
    "Memory ECC error corrected on <*>",
    "Node rebooted successfully",
]

HDFS_COMPONENTS = ["dfs.DataNode", "dfs.FSNamesystem", "dfs.DataNode$DataXceiver", "hdfs.DFSClient"]
HDFS_MESSAGES = [
    "Receiving block blk_-1234567890 src: /10.0.0.1:50010 dest: /10.0.0.2:50010",
    "BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.0.0.3:50010 is added",
    "Served block blk_9876543210 to /10.0.0.4",
    "Deleting block blk_1111111111 file /tmp/data.txt",
    "PacketResponder 0 for block blk_2222222222 terminating",
]
HDFS_TEMPLATES = [
    "Receiving block <*> src: <*> dest: <*>",
    "BLOCK* NameSystem.addStoredBlock: blockMap updated: <*> is added",
    "Served block <*> to <*>",
    "Deleting block <*> file <*>",
    "PacketResponder <*> for block <*> terminating",
]

HADOOP_COMPONENTS = ["RMAppManager", "LeafQueue", "FairScheduler", "TaskAttempt"]
HADOOP_MESSAGES = [
    "Storing application with id application_1234_0001",
    "Application added - appId: application_1234_0002 user: hdfs queue: default",
    "Scheduling resource request for application_1234_0003",
    "Completed container container_1234_0001_01_000001",
    "Task attempt_1234_0002_m_000001_0 done",
]
HADOOP_TEMPLATES = [
    "Storing application with id <*>",
    "Application added - appId: <*> user: <*> queue: <*>",
    "Scheduling resource request for <*>",
    "Completed container <*>",
    "Task <*> done",
]

SPARK_COMPONENTS = ["SparkContext", "DAGScheduler", "TaskSetManager", "BlockManager", "Executor"]
SPARK_MESSAGES = [
    "Starting job 0",
    "Got job 1 with 4 output partitions",
    "Submitting 4 missing tasks from ShuffleMapStage 2",
    "Finished task 0.0 in stage 1.0 (TID 5) in 120 ms",
    "Block broadcast_1 stored as values in memory (estimated size 5.7 KB)",
]
SPARK_TEMPLATES = [
    "Starting job <*>",
    "Got job <*> with <*> output partitions",
    "Submitting <*> missing tasks from <*>",
    "Finished task <*> in stage <*> (TID <*>) in <*> ms",
    "Block <*> stored as values in memory (estimated size <*>)",
]


def _gen_linux_row(i: int) -> str:
    idx = i % len(LINUX_MESSAGES)
    month = random.choice(MONTHS)
    day = random.randint(1, 28)
    time = f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
    comp = random.choice(LINUX_COMPONENTS)
    pid = random.randint(1000, 65000)
    eid = f"E{idx + 1}"
    return f"{i + 1},{month},{day},{time},combo,{comp},{pid},{LINUX_MESSAGES[idx]},{eid},{LINUX_TEMPLATES[idx]}"


def _gen_hpc_row(i: int) -> str:
    idx = i % len(HPC_MESSAGES)
    log_id = f"logid_{i}"
    node = f"node-{random.randint(1, 500)}"
    comp = random.choice(HPC_COMPONENTS)
    state = random.choice(HPC_STATES)
    timestamp = 1077804742 + i * 60
    flag = str(random.choice([0, 0, 0, 1]))
    eid = f"E{idx + 1}"
    return f"{i + 1},{log_id},{node},{comp},{state},{timestamp},{flag},{HPC_MESSAGES[idx]},{eid},{HPC_TEMPLATES[idx]}"


def _gen_hdfs_row(i: int) -> str:
    idx = i % len(HDFS_MESSAGES)
    date = f"08{random.randint(10,12):02d}{random.randint(10,30):02d}"
    time = f"{random.randint(0,23):02d}{random.randint(0,59):02d}{random.randint(0,59):02d}"
    pid = random.randint(1000, 9999)
    level = random.choices(LEVELS, weights=LEVEL_WEIGHTS)[0]
    comp = random.choice(HDFS_COMPONENTS)
    eid = f"E{idx + 1}"
    return f"{i + 1},{date},{time},{pid},{level},{comp},{HDFS_MESSAGES[idx]},{eid},{HDFS_TEMPLATES[idx]}"


def _gen_hadoop_row(i: int) -> str:
    idx = i % len(HADOOP_MESSAGES)
    date = f"2015-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    time = f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
    level = random.choices(LEVELS, weights=LEVEL_WEIGHTS)[0]
    process = f"main-EventThread"
    comp = random.choice(HADOOP_COMPONENTS)
    eid = f"E{idx + 1}"
    return f"{i + 1},{date},{time},{level},{process},{comp},{HADOOP_MESSAGES[idx]},{eid},{HADOOP_TEMPLATES[idx]}"


def _gen_spark_row(i: int) -> str:
    idx = i % len(SPARK_MESSAGES)
    date = f"17/{random.randint(1,12):02d}/{random.randint(1,28):02d}"
    time = f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
    level = random.choices(LEVELS, weights=LEVEL_WEIGHTS)[0]
    comp = random.choice(SPARK_COMPONENTS)
    eid = f"E{idx + 1}"
    return f"{i + 1},{date},{time},{level},{comp},{SPARK_MESSAGES[idx]},{eid},{SPARK_TEMPLATES[idx]}"


_ROW_GENERATORS = {
    "Linux": _gen_linux_row,
    "HPC": _gen_hpc_row,
    "HDFS": _gen_hdfs_row,
    "Hadoop": _gen_hadoop_row,
    "Spark": _gen_spark_row,
}

_TEMPLATE_DATA = {
    "Linux": (LINUX_TEMPLATES,),
    "HPC": (HPC_TEMPLATES,),
    "HDFS": (HDFS_TEMPLATES,),
    "Hadoop": (HADOOP_TEMPLATES,),
    "Spark": (SPARK_TEMPLATES,),
}


def _gen_templates(system: str, num_rows: int) -> list[str]:
    templates = _TEMPLATE_DATA[system][0]
    rows = []
    for idx, t in enumerate(templates):
        occurrences = max(1, num_rows // len(templates))
        rows.append(f"E{idx + 1},{t},{occurrences}")
    return rows


def generate_all(loghub_dir: Path, num_rows: int = 50) -> None:
    """Generate synthetic structured and template CSVs for all 5 systems."""
    for system, schema in SYSTEM_SCHEMAS.items():
        sys_dir = loghub_dir / system
        sys_dir.mkdir(parents=True, exist_ok=True)

        gen_fn = _ROW_GENERATORS[system]
        structured_rows = [schema["structured_cols"]]
        for i in range(num_rows):
            structured_rows.append(gen_fn(i))
        structured_file = sys_dir / f"{system}_2k.log_structured.csv"
        structured_file.write_text("\n".join(structured_rows) + "\n")

        template_rows = [schema["template_cols"]]
        template_rows.extend(_gen_templates(system, num_rows))
        template_file = sys_dir / f"{system}_2k.log_templates.csv"
        template_file.write_text("\n".join(template_rows) + "\n")

        print(f"  Created {system}: {num_rows} structured rows, {len(template_rows) - 1} templates")

    print(f"DS2 seed data saved to {loghub_dir}")


def main():
    try:
        from src.config.paths import get_ds2_raw_dir
        raw_dir = get_ds2_raw_dir()
    except ImportError:
        raw_dir = PROJECT_ROOT / "data" / "raw" / "ds2_loghub"

    print("Generating DS2 (LogHub) seed data...")
    generate_all(raw_dir / "loghub")
    return True


if __name__ == "__main__":
    main()
