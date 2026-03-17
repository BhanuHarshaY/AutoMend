"""Seed the prompts DB with user intents for pipeline testing.

Default mode seeds 15 prompts.  Expanded mode ('full') seeds ~100 diverse
MLOps intents for richer synthetic data generation.

Available tools are loaded from config/available_tools.json at pipeline run
time; only user intents are stored here."""
import sqlite3
import sys
from pathlib import Path

DS4_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = DS4_ROOT.parent.parent
sys.path.insert(0, str(DS4_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_4_synthetic.src.data import db_ops
from src.config.paths import get_ds4_prompts_db

DB_PATH = get_ds4_prompts_db()

TOOL_CONTEXT_PLACEHOLDER = ""

DEFAULT_INTENTS = [
    "Fix the latency on the fraud model.",
    "Scale up the recommendation service to handle more load.",
    "Restart the inference pod for the churn model.",
    "Fix the memory leak on the fraud model.",
    "Increase replicas for the API server.",
    "The model server is OOM; restart it and scale down if needed.",
    "Reduce replicas for the batch scoring job to save cost.",
    "Restart all pods for the training pipeline.",
    "Scale the feature store service to 5 replicas.",
    "The A/B test model is stuck; restart its deployment.",
    "Double the replicas for the real-time predictor.",
    "Fix high CPU on the data ingestion pod by restarting it.",
    "Scale down the dev environment to 1 replica.",
    "Restart the model monitoring service.",
    "Handle traffic spike: scale the serving layer to 10 replicas.",
]

EXPANDED_INTENTS = DEFAULT_INTENTS + [
    "Roll back the sentiment model to the previous version.",
    "Deploy the new fraud detection model v2.3 to staging.",
    "Scale the embedding service horizontally to handle 10k QPS.",
    "Investigate and fix the data pipeline lag for feature store updates.",
    "Restart the MLflow tracking server; experiments are failing to log.",
    "Set up autoscaling for the image classification endpoint.",
    "Migrate the recommendation model from GPU to CPU serving.",
    "Update the model registry to promote churn-v4 to production.",
    "Run a canary deployment for the new ranking model.",
    "Debug the OOM crash on the NLP preprocessing pod.",
    "Scale the vector database cluster from 3 to 5 nodes.",
    "Restart the Airflow scheduler; DAGs are not being picked up.",
    "Fix the broken data validation step in the training pipeline.",
    "Deploy a shadow model alongside the production fraud detector.",
    "Scale down non-critical batch jobs during peak serving hours.",
    "Update the TensorFlow Serving container to version 2.15.",
    "Restart the Prometheus monitoring stack after config changes.",
    "Set up a blue-green deployment for the search ranking model.",
    "Fix the certificate expiry on the model serving endpoint.",
    "Scale the preprocessing workers to 20 for the nightly batch job.",
    "Restart the Redis cache used by the real-time feature store.",
    "Deploy the updated tokenizer model to the NLP pipeline.",
    "Investigate high latency on the product recommendation API.",
    "Roll back the A/B test config to the previous experiment.",
    "Scale the GPU training cluster to 8 nodes for the large model.",
    "Fix the broken webhook notification for pipeline failures.",
    "Restart the Grafana dashboard server.",
    "Deploy the new anomaly detection model to the edge cluster.",
    "Scale the log aggregation pipeline to handle increased volume.",
    "Fix the authentication issue on the model registry API.",
    "Set up rate limiting on the prediction API to prevent abuse.",
    "Restart the data labeling service after database migration.",
    "Deploy a new version of the feature engineering pipeline.",
    "Scale the inference cluster GPU nodes from 2 to 4.",
    "Fix the disk full issue on the model artifact storage.",
    "Restart the experiment tracking service after upgrade.",
    "Deploy the text generation model to a dedicated GPU node pool.",
    "Investigate the data drift alert on the fraud detection model.",
    "Scale the API gateway to handle the holiday traffic surge.",
    "Fix the broken CI/CD pipeline for model deployment.",
    "Restart the distributed training job that hung at epoch 45.",
    "Deploy the object detection model to the IoT edge devices.",
    "Scale the batch prediction job to process 1M records.",
    "Fix the model versioning conflict in the registry.",
    "Restart the feature extraction service after memory upgrade.",
    "Deploy the updated bias detection module to all pipelines.",
    "Scale the monitoring infrastructure to cover new model endpoints.",
    "Fix the data schema mismatch between training and serving.",
    "Restart the automated retraining trigger service.",
    "Deploy the optimized ONNX model to replace the PyTorch version.",
    "Scale the data preprocessing queue workers from 5 to 15.",
    "Fix the failed model evaluation step in the CI pipeline.",
    "Restart the model explainability service after API changes.",
    "Deploy a new custom metric exporter for model performance.",
    "Scale the training data pipeline to ingest from 3 new sources.",
    "Fix the broken model rollback automation.",
    "Restart the cost optimization service for cloud GPU usage.",
    "Deploy the federated learning coordinator to the cluster.",
    "Scale the real-time scoring service for the flash sale event.",
    "Fix the missing model metadata in the experiment tracker.",
    "Restart the data quality monitoring job after schema update.",
    "Deploy the distilled model to reduce inference costs by 40%.",
    "Scale the hyperparameter tuning job to use 16 parallel trials.",
    "Fix the broken Slack alerting for model degradation.",
    "Restart the model A/B testing framework after an update.",
    "Deploy the multi-armed bandit model for ad ranking.",
    "Scale the online learning pipeline to process streaming data.",
    "Fix the dependency conflict in the model serving container.",
    "Restart the centralized logging service for ML pipelines.",
    "Deploy the new speech-to-text model to the voice assistant.",
    "Scale the ETL pipeline for the quarterly data refresh.",
    "Fix the model cache invalidation issue on serving nodes.",
    "Restart the automated model retraining scheduler.",
    "Deploy the updated data augmentation pipeline.",
    "Scale the graph neural network training to 4 GPU nodes.",
    "Fix the broken model performance dashboard.",
    "Restart the feature flag service for model experiments.",
    "Deploy the reinforcement learning agent to the simulation env.",
    "Scale the model serving infrastructure for multi-region.",
    "Fix the training data sampling bias in the pipeline.",
    "Restart the model comparison service after DB migration.",
    "Deploy the time-series forecasting model to production.",
]


def seed_prompts(
    prompt_set: str = "default",
    prompt_count: int | None = None,
) -> int:
    """Seed the prompts database.

    Parameters
    ----------
    prompt_set : str
        ``"default"`` (15 intents) or ``"expanded"`` (~100 intents).
    prompt_count : int | None
        Override: exact number of prompts to insert.  ``None`` means use all
        from the chosen set.

    Returns
    -------
    int
        Number of prompts inserted.
    """
    intents = EXPANDED_INTENTS if prompt_set == "expanded" else DEFAULT_INTENTS
    if prompt_count is not None and prompt_count > 0:
        intents = intents[:prompt_count]

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        db_ops.setup_db(conn)
        existing = conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]
        if existing >= len(intents):
            print(f"DB already has {existing} prompts, skipping seed.")
            return existing
        for intent in intents:
            conn.execute(
                "INSERT OR IGNORE INTO prompts (user_intent, tool_context, processed) VALUES (?, ?, 0)",
                (intent, TOOL_CONTEXT_PLACEHOLDER),
            )
        conn.commit()
        final = conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]
        print(f"Seeded {final} prompts into {DB_PATH}")
        return final
    finally:
        conn.close()


def main():
    seed_prompts()


if __name__ == "__main__":
    main()
