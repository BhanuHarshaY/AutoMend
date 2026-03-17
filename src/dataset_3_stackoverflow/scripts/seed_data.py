"""
Seed script for DS3 (StackOverflow) - generates sample Q&A CSV files for E2E testing.

Creates 2 CSV files with realistic DevOps/MLOps Q&A data:
- Stack_Qns_pl.csv: Sample questions with Id, Title, Body, Tags, Score, ViewCount, AcceptedAnswerId
- Stack_Ans_pl.csv: Sample answers with AnswerId, QuestionId, AnswerBody, AnswerScore

Usage:
    python -m src.dataset_3_stackoverflow.scripts.seed_data
    python src/dataset_3_stackoverflow/scripts/seed_data.py
"""

import csv
import random
import sys
from pathlib import Path

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to use centralized paths
try:
    from src.config.paths import get_ds3_raw_dir, DATA_ROOT
    RAW_DIR = get_ds3_raw_dir()
    EXTERNAL_DIR = DATA_ROOT / "external"
except ImportError:
    RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ds3_stackoverflow"
    EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

# Seed for reproducibility
random.seed(42)

# Output files (external CSV format expected by data_acquisition.py)
QUESTIONS_FILE = EXTERNAL_DIR / "Stack_Qns_pl.csv"
ANSWERS_FILE = EXTERNAL_DIR / "Stack_Ans_pl.csv"

# Sample DevOps/MLOps Q&A data
SAMPLE_QA = [
    {
        "title": "How to fix CrashLoopBackOff in Kubernetes pod?",
        "body": "<p>My Kubernetes pod keeps entering CrashLoopBackOff state. The logs show the container starts but then exits immediately with code 1. I've checked the resource limits and they seem adequate.</p><pre><code>kubectl get pods\nNAME                     READY   STATUS             RESTARTS   AGE\nmy-app-6b7d6c9f5-xyz12   0/1     CrashLoopBackOff   5          10m</code></pre>",
        "tags": "<kubernetes><docker><crashloopbackoff>",
        "answer": "<p>CrashLoopBackOff usually indicates your container is crashing repeatedly. Here are the steps to debug:</p><ol><li>Check logs: <code>kubectl logs my-app-6b7d6c9f5-xyz12 --previous</code></li><li>Verify entry point is correct in Dockerfile</li><li>Check environment variables and config maps</li><li>Ensure the application can connect to required services</li></ol><p>Most common causes are misconfigured environment variables or missing dependencies.</p>",
    },
    {
        "title": "Terraform state lock stuck - how to force unlock?",
        "body": "<p>I was running terraform apply and it crashed midway. Now I get this error:</p><pre><code>Error: Error locking state: Error acquiring the state lock</code></pre><p>The DynamoDB table shows the lock is still held. Is it safe to force unlock?</p>",
        "tags": "<terraform><aws><infrastructure-as-code>",
        "answer": "<p>You can force unlock the state using:</p><pre><code>terraform force-unlock LOCK_ID</code></pre><p>Get the LOCK_ID from the error message or DynamoDB table. Before doing this, make sure no other terraform operations are actually running. After unlocking, run <code>terraform plan</code> first to verify state consistency before applying any changes.</p>",
    },
    {
        "title": "OOMKilled - Kubernetes container keeps getting killed",
        "body": "<p>My ML training job keeps getting OOMKilled. I've set memory limits to 8Gi but the pod still gets terminated.</p><pre><code>Last State:     Terminated\n  Reason:       OOMKilled\n  Exit Code:    137</code></pre>",
        "tags": "<kubernetes><memory><oomkilled><mlops>",
        "answer": "<p>Exit code 137 confirms OOM. Your application is using more memory than the limit. Solutions:</p><ul><li>Increase memory limit: <code>resources.limits.memory: 16Gi</code></li><li>Optimize your code to use less memory (batch processing, generators)</li><li>Use memory profiling tools to find leaks</li><li>For ML workloads, reduce batch size or use gradient checkpointing</li></ul>",
    },
    {
        "title": "Docker image pull fails with unauthorized error",
        "body": "<p>When deploying to Kubernetes, I get ImagePullBackOff with this error:</p><pre><code>Failed to pull image: unauthorized: authentication required</code></pre><p>The image exists in our private ECR registry.</p>",
        "tags": "<kubernetes><docker><aws><ecr>",
        "answer": "<p>For ECR, you need to create an image pull secret:</p><pre><code>kubectl create secret docker-registry ecr-secret \\\n  --docker-server=123456789.dkr.ecr.us-east-1.amazonaws.com \\\n  --docker-username=AWS \\\n  --docker-password=$(aws ecr get-login-password)</code></pre><p>Then reference it in your pod spec:</p><pre><code>imagePullSecrets:\n  - name: ecr-secret</code></pre>",
    },
    {
        "title": "CUDA out of memory error during PyTorch training",
        "body": "<p>Getting CUDA OOM during training even with batch_size=1:</p><pre><code>RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB</code></pre><p>GPU has 16GB VRAM, model size should fit.</p>",
        "tags": "<pytorch><cuda><gpu><deep-learning>",
        "answer": "<p>Try these techniques:</p><ol><li>Enable gradient checkpointing: <code>model.gradient_checkpointing_enable()</code></li><li>Use mixed precision: <code>torch.cuda.amp.autocast()</code></li><li>Clear cache periodically: <code>torch.cuda.empty_cache()</code></li><li>Reduce model size or use model parallelism</li><li>Check for memory leaks in data loading</li></ol>",
    },
    {
        "title": "Prometheus not scraping targets - connection refused",
        "body": "<p>Prometheus shows all targets as DOWN with 'connection refused'. Services are running and accessible via curl from other pods.</p>",
        "tags": "<prometheus><kubernetes><monitoring>",
        "answer": "<p>Common causes:</p><ul><li>Check if pods have correct annotations for scraping</li><li>Verify network policies allow Prometheus access</li><li>Ensure service ports match scrape config</li><li>Check if metrics endpoint path is correct (usually /metrics)</li></ul><pre><code>annotations:\n  prometheus.io/scrape: 'true'\n  prometheus.io/port: '8080'</code></pre>",
    },
    {
        "title": "Airflow DAG not showing up in UI",
        "body": "<p>I created a new DAG file but it doesn't appear in the Airflow web UI. No errors in the logs either.</p>",
        "tags": "<airflow><python><dag>",
        "answer": "<p>Common reasons DAGs don't appear:</p><ol><li>Syntax error in DAG file - test with <code>python your_dag.py</code></li><li>DAG file not in the configured dags_folder</li><li>Import error in the DAG file</li><li>DAG schedule_interval is in the past with catchup=False</li><li>Check airflow.cfg for dag_dir_list_interval</li></ol>",
    },
    {
        "title": "Helm upgrade fails with resource already exists",
        "body": "<p>Running helm upgrade gives error: cannot patch - field is immutable</p><pre><code>Error: UPGRADE FAILED: cannot patch \"my-service\" with kind Service</code></pre>",
        "tags": "<kubernetes><helm><deployment>",
        "answer": "<p>Some Kubernetes fields are immutable and can't be changed via upgrade. Options:</p><ol><li>Delete and recreate: <code>helm uninstall myrelease && helm install</code></li><li>Use --force flag: <code>helm upgrade --force myrelease ./chart</code></li><li>Manually delete the resource: <code>kubectl delete svc my-service</code></li></ol><p>Be careful with --force in production as it causes downtime.</p>",
    },
    {
        "title": "TensorFlow GPU not detected on Kubernetes",
        "body": "<p>TensorFlow shows no GPUs available when running in Kubernetes pod, but nvidia-smi works fine:</p><pre><code>tf.config.list_physical_devices('GPU')\n[]</code></pre>",
        "tags": "<tensorflow><gpu><kubernetes><nvidia>",
        "answer": "<p>Ensure proper GPU resource requests:</p><pre><code>resources:\n  limits:\n    nvidia.com/gpu: 1</code></pre><p>Also verify:</p><ul><li>NVIDIA device plugin is installed on the cluster</li><li>Container image has correct CUDA version</li><li>tensorflow-gpu is installed (or tf 2.x+ with GPU support)</li><li>LD_LIBRARY_PATH includes CUDA libs</li></ul>",
    },
    {
        "title": "MLflow tracking server connection timeout",
        "body": "<p>MLflow experiments fail with connection timeout to tracking server:</p><pre><code>requests.exceptions.ConnectionError: Connection to mlflow.internal timed out</code></pre>",
        "tags": "<mlflow><mlops><networking>",
        "answer": "<p>Debug steps:</p><ol><li>Check if MLflow server is running: <code>kubectl get pods -l app=mlflow</code></li><li>Verify service DNS: <code>nslookup mlflow.internal</code></li><li>Test connectivity: <code>curl http://mlflow.internal:5000/health</code></li><li>Check for network policies blocking traffic</li><li>Verify MLFLOW_TRACKING_URI is correct</li></ol>",
    },
    {
        "title": "Jenkins pipeline fails with git authentication error",
        "body": "<p>Jenkins pipeline fails at git checkout stage with permission denied error despite having credentials configured.</p>",
        "tags": "<jenkins><git><cicd>",
        "answer": "<p>Common fixes:</p><ul><li>Regenerate credentials/SSH keys and update in Jenkins</li><li>For SSH: ensure known_hosts is configured</li><li>For HTTPS: use credential binding in pipeline</li></ul><pre><code>withCredentials([usernamePassword(credentialsId: 'git-creds', usernameVariable: 'GIT_USER', passwordVariable: 'GIT_PASS')]) {\n  sh 'git clone https://$GIT_USER:$GIT_PASS@github.com/repo'\n}</code></pre>",
    },
    {
        "title": "Redis connection pool exhausted in production",
        "body": "<p>Getting 'connection pool exhausted' errors from Redis during high traffic. Currently using default connection settings.</p>",
        "tags": "<redis><python><performance>",
        "answer": "<p>Increase pool size and implement proper connection handling:</p><pre><code>pool = redis.ConnectionPool(\n    host='redis', port=6379,\n    max_connections=50,\n    socket_timeout=5,\n    socket_connect_timeout=5\n)\nredis_client = redis.Redis(connection_pool=pool)</code></pre><p>Also ensure you're not creating new clients for each request and properly closing connections.</p>",
    },
    {
        "title": "Grafana dashboard shows no data for metrics",
        "body": "<p>Grafana queries return 'No data' even though Prometheus has the metrics. Query works in Prometheus UI but not in Grafana.</p>",
        "tags": "<grafana><prometheus><monitoring>",
        "answer": "<p>Check these common issues:</p><ol><li>Time range - ensure dashboard time range includes data</li><li>Data source - verify Prometheus URL is correct in Grafana</li><li>Query syntax - Grafana uses PromQL, verify query format</li><li>Variable interpolation - check if template variables are correct</li><li>Browser cache - try hard refresh (Ctrl+Shift+R)</li></ol>",
    },
    {
        "title": "AWS EKS node group scaling issues",
        "body": "<p>EKS cluster autoscaler doesn't scale up when pods are pending. Cluster autoscaler logs show no errors.</p>",
        "tags": "<aws><eks><kubernetes><autoscaling>",
        "answer": "<p>Verify autoscaler configuration:</p><ol><li>Check node group min/max settings allow scaling</li><li>Verify autoscaler has correct IAM permissions</li><li>Check if pending pods have resource requests defined</li><li>Look for node selector/affinity preventing scheduling</li></ol><pre><code>kubectl logs -n kube-system deployment/cluster-autoscaler | grep -i scale</code></pre>",
    },
    {
        "title": "DVC push fails with remote storage authentication",
        "body": "<p>DVC push fails with authentication error to S3 remote:</p><pre><code>ERROR: failed to push - Unable to locate credentials</code></pre>",
        "tags": "<dvc><mlops><aws><s3>",
        "answer": "<p>Configure AWS credentials for DVC:</p><ol><li>Using environment variables:<br><code>export AWS_ACCESS_KEY_ID=xxx</code><br><code>export AWS_SECRET_ACCESS_KEY=xxx</code></li><li>Using AWS profile:<br><code>dvc remote modify myremote profile myprofile</code></li><li>Using instance role (EC2/EKS):<br>Ensure IAM role has S3 permissions</li></ol><p>Verify with: <code>aws s3 ls s3://your-bucket</code></p>",
    },
]


def generate_questions(num_rows: int = 50) -> list[dict]:
    """Generate sample questions CSV rows."""
    questions = []
    
    for i in range(num_rows):
        sample = SAMPLE_QA[i % len(SAMPLE_QA)]
        
        question_id = 1000000 + i
        accepted_answer_id = 2000000 + i
        score = random.randint(5, 500)
        view_count = random.randint(100, 50000)
        creation_date = f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}T{random.randint(0,23):02d}:00:00Z"
        
        questions.append({
            "Id": question_id,
            "Title": sample["title"],
            "Body": sample["body"],
            "Tags": sample["tags"],
            "Score": score,
            "ViewCount": view_count,
            "AcceptedAnswerId": accepted_answer_id,
            "CreationDate": creation_date,
        })
    
    return questions


def generate_answers(questions: list[dict]) -> list[dict]:
    """Generate sample answers CSV rows matching the questions."""
    answers = []
    
    for i, q in enumerate(questions):
        sample = SAMPLE_QA[i % len(SAMPLE_QA)]
        
        answers.append({
            "AnswerId": q["AcceptedAnswerId"],
            "QuestionId": q["Id"],
            "AnswerBody": sample["answer"],
            "AnswerScore": random.randint(3, 200),
        })
    
    return answers


def main(num_rows: int | None = None):
    """Generate all seed data files for DS3.

    Parameters
    ----------
    num_rows : int | None
        Number of Q&A rows.  Defaults to 50.
    """
    n = num_rows if num_rows is not None else 50
    print(f"Generating DS3 (StackOverflow) seed data ({n} rows)...")
    print(f"Output directory: {EXTERNAL_DIR}")

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    questions = generate_questions(n)
    
    # Write questions CSV
    with open(QUESTIONS_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Id", "Title", "Body", "Tags", "Score", "ViewCount", "AcceptedAnswerId", "CreationDate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(questions)
    print(f"  Created: {QUESTIONS_FILE.name} ({len(questions)} rows)")
    
    # Generate and write answers CSV
    answers = generate_answers(questions)
    
    with open(ANSWERS_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["AnswerId", "QuestionId", "AnswerBody", "AnswerScore"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(answers)
    print(f"  Created: {ANSWERS_FILE.name} ({len(answers)} rows)")
    
    print(f"\nDS3 seed data generation complete!")
    print(f"Files saved to: {EXTERNAL_DIR}")
    return True


if __name__ == "__main__":
    main()
