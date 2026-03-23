"""
workflow_schema.py

Pydantic v2 models defining the two valid output shapes for Model 2.

The model produces one of two shapes:

  Shape A — Tool workflow (model decided to call one or more tools):
    {
      "workflow": {
        "steps": [
          {"tool": "calculate_age", "params": {"birth_date": "1990-05-15"}}
        ]
      }
    }

  Shape B — Natural language response (no applicable tool):
    {
      "workflow": {"steps": []},
      "message": "I'm sorry, I cannot assist with that."
    }

Rules encoded in the models:
  - "workflow" key must exist at root
  - "steps" must be a list (empty or non-empty)
  - Each step must have "tool" (non-empty string) and "params" (dict)
  - Shape A: steps is non-empty, no "message" key
  - Shape B: steps is empty, "message" is a non-empty string
  - No extra top-level keys on either shape (extra="forbid")
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


# ---------------------------------------------------------------------------
# Step-level model
# ---------------------------------------------------------------------------

class WorkflowStep(BaseModel):
    tool: str
    params: dict[str, Any]
    model_config = ConfigDict(extra="forbid")

    @field_validator("tool")
    @classmethod
    def tool_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("tool name must be a non-empty string")
        return v


# ---------------------------------------------------------------------------
# Workflow container
# ---------------------------------------------------------------------------

class Workflow(BaseModel):
    steps: list[WorkflowStep]
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Shape A — tool call (steps non-empty, no message)
# ---------------------------------------------------------------------------

class ToolWorkflowResponse(BaseModel):
    workflow: Workflow
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def steps_must_be_nonempty(self) -> "ToolWorkflowResponse":
        if not self.workflow.steps:
            raise ValueError(
                "ToolWorkflowResponse requires at least one step; "
                "use MessageWorkflowResponse for empty steps"
            )
        return self


# ---------------------------------------------------------------------------
# Shape B — natural language response (steps empty, message required)
# ---------------------------------------------------------------------------

class MessageWorkflowResponse(BaseModel):
    workflow: Workflow
    message: str
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def steps_must_be_empty(self) -> "MessageWorkflowResponse":
        if self.workflow.steps:
            raise ValueError(
                "MessageWorkflowResponse must have empty steps; "
                "use ToolWorkflowResponse for non-empty steps"
            )
        return self

    @field_validator("message")
    @classmethod
    def message_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("message must be a non-empty string")
        return v


# ---------------------------------------------------------------------------
# Output shape enum (for shape comparisons without full validation)
# ---------------------------------------------------------------------------

class OutputShape:
    TOOL = "tool"
    MESSAGE = "message"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_response(text: str) -> tuple[ToolWorkflowResponse | MessageWorkflowResponse | None, str | None]:
    """
    Try to parse a raw generated string into a validated WorkflowResponse.

    Tries ToolWorkflowResponse first, then MessageWorkflowResponse.

    Returns:
        (model_instance, None)   on success
        (None, error_reason_str) on failure
    """
    if not text or not text.strip():
        return None, "empty_output"

    try:
        data = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None, "json_parse_failed"

    if not isinstance(data, dict):
        return None, "not_a_dict"

    # Try Shape A first (tool workflow)
    try:
        return ToolWorkflowResponse.model_validate(data), None
    except Exception:
        pass

    # Try Shape B (message response)
    try:
        return MessageWorkflowResponse.model_validate(data), None
    except Exception as e:
        return None, f"schema_invalid: {_summarize_pydantic_error(e)}"


def infer_shape(text: str) -> str:
    """
    Infer the output shape from raw text WITHOUT full Pydantic validation.
    Used to compute correct_shape_rate even when schema validation fails.

    Returns OutputShape.TOOL, OutputShape.MESSAGE, or OutputShape.UNKNOWN.
    """
    if not text or not text.strip():
        return OutputShape.UNKNOWN

    try:
        data = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return OutputShape.UNKNOWN

    if not isinstance(data, dict):
        return OutputShape.UNKNOWN

    workflow = data.get("workflow")
    if not isinstance(workflow, dict):
        return OutputShape.UNKNOWN

    steps = workflow.get("steps")
    if not isinstance(steps, list):
        return OutputShape.UNKNOWN

    if steps:
        return OutputShape.TOOL
    elif "message" in data:
        return OutputShape.MESSAGE
    else:
        return OutputShape.UNKNOWN


def _summarize_pydantic_error(exc: Exception) -> str:
    """Extract a short error summary from a Pydantic ValidationError."""
    try:
        errors = exc.errors()  # type: ignore[attr-defined]
        types = list({e["type"] for e in errors})
        return ", ".join(types[:3])
    except Exception:
        return str(exc)[:80]
