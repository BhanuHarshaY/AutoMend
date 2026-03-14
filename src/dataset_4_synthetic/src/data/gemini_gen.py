"""Gemini 2.5 Flash API calls for synthetic workflow generation.

Supports both sequential generation and Ray-actor-based parallel generation
with asyncio for high throughput while respecting rate limits.
"""
import asyncio
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from google import genai

from schemas.workflow_schema import Workflow
from data.pipeline_logger import get_logger

logger = get_logger(__name__)
GEMINI_MODEL = "gemini-2.5-flash"

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY", "dummy-replace-with-real-key")

WORKFLOW_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "integer"},
                    "tool": {"type": "string"},
                    "params": {
                        "type": "object",
                        "properties": {
                            "deployment": {"type": "string"},
                            "replicas": {"type": "integer"},
                            "pod": {"type": "string"},
                        },
                    },
                },
                "required": ["step_id", "tool", "params"],
            },
        },
    },
    "required": ["steps"],
}


def generate_workflow(user_intent: str, available_tools: list[str] | str) -> Workflow:
    """Call Gemini to generate a structured workflow JSON and return a validated Workflow."""
    logger.info("Generating workflow for intent: %s", user_intent[:50] + "..." if len(user_intent) > 50 else user_intent)
    if isinstance(available_tools, list):
        tools_str = ", ".join(available_tools)
    else:
        tools_str = available_tools
    client = genai.Client(api_key=GOOGLE_API_KEY)
    prompt = (
        f"Given the user intent and available tools, output a JSON workflow (steps with step_id, tool, params). "
        f"User intent: {user_intent}. Available tools: {tools_str}. "
        "Output only valid JSON with a 'steps' array; each step has step_id (int), tool (string), params (object with optional deployment, replicas, pod)."
    )
    config = {
        "response_mime_type": "application/json",
        "response_json_schema": WORKFLOW_JSON_SCHEMA,
    }
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=config,
    )
    workflow = Workflow.model_validate_json(response.text)
    logger.info("Workflow generated: %d steps", len(workflow.steps))
    return workflow


# ---------------------------------------------------------------------------
# Ray Actor for parallel generation with asyncio
# ---------------------------------------------------------------------------

def _make_gemini_worker_class():
    """Factory to avoid importing ray at module level."""
    import ray

    @ray.remote
    class GeminiWorker:
        def __init__(self, api_key: str, max_concurrent: int = 5):
            self.client = genai.Client(api_key=api_key)
            self.semaphore = asyncio.Semaphore(max_concurrent)

        async def _generate_one(self, user_intent: str, tools_str: str) -> dict:
            async with self.semaphore:
                prompt = (
                    f"Given the user intent and available tools, output a JSON workflow. "
                    f"User intent: {user_intent}. Available tools: {tools_str}. "
                    "Output only valid JSON with a 'steps' array."
                )
                config = {
                    "response_mime_type": "application/json",
                    "response_json_schema": WORKFLOW_JSON_SCHEMA,
                }
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=config,
                )
                workflow = Workflow.model_validate_json(response.text)
                return workflow.model_dump()

        async def generate_batch(self, prompts: list[dict], tools_str: str) -> list[dict]:
            tasks = [
                self._generate_one(p["user_intent"], tools_str)
                for p in prompts
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            out = []
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    logger.error("Prompt %d failed: %s", i, r)
                    out.append(None)
                else:
                    out.append(r)
            return out

    return GeminiWorker


def generate_workflows_parallel(
    prompts: list[dict],
    tools: list[str],
    num_workers: int = 4,
    max_concurrent: int = 5,
) -> list[Optional[dict]]:
    """Generate workflows in parallel using Ray actors with async Gemini calls."""
    import ray
    from src.config.ray_config import init_ray

    init_ray()
    GeminiWorker = _make_gemini_worker_class()
    tools_str = ", ".join(tools)
    api_key = GOOGLE_API_KEY

    workers = [GeminiWorker.remote(api_key, max_concurrent) for _ in range(num_workers)]

    batch_size = max(1, len(prompts) // num_workers)
    batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]

    futures = []
    for i, batch in enumerate(batches):
        worker = workers[i % num_workers]
        futures.append(worker.generate_batch.remote(batch, tools_str))

    all_results = ray.get(futures)
    flat = []
    for batch_result in all_results:
        flat.extend(batch_result)
    return flat
