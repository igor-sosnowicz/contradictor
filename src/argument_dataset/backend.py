"""
LM Studio agents built with pydantic-ai.

Every request is standalone (no conversation memory).
"""

import asyncio
from pathlib import Path
from string import Template

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from src.argument_dataset.output_models import (
    ExtractedArgument,
    ExtractionVerdict,
    LinkVerdict,
)
from src.configuration import config as global_config
from src.data_models.data_models import InterpretativeFrame

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


# TODO(nsjg): move to structured_output_models.py  # noqa: FIX002, TD003
class ExtractionBatch(BaseModel):
    """Extractor output: all arguments found in one chunk."""

    arguments: list[ExtractedArgument] = Field(default_factory=list)


def render_prompt(filename: str) -> str:
    """Load a prompt template, filling in the guide and frame list."""
    template = (PROMPTS_DIR / filename).read_text(encoding="utf-8")
    guide = (PROMPTS_DIR / "argument_type_guide.md").read_text(encoding="utf-8")
    frames = ", ".join(f.value for f in InterpretativeFrame)
    return Template(template).substitute(
        ARGUMENT_TYPE_GUIDE=guide.strip(), FRAMES=frames
    )


def _model(name: str) -> OpenAIChatModel:
    """
    Point an OpenAI-compatible model at the local LM Studio server.

    ``PromptedOutput`` keeps the model in plain-text mode: this LM Studio
    build rejects ``json_object`` response formats, and its constrained
    ``json_schema`` decoding degrades small models to empty outputs.
    """
    provider = OpenAIProvider(
        base_url=f"http://{global_config.lm_studio_api_base_url}/v1",
        api_key=global_config.lm_studio_api_key,
    )
    profile = ModelProfile(supports_json_object_output=False)
    return OpenAIChatModel(name, provider=provider, profile=profile)


def make_agents(
    extractor_name: str, judge_name: str
) -> tuple[
    Agent[None, ExtractionBatch],
    Agent[None, ExtractionVerdict],
    Agent[None, LinkVerdict],
]:
    """Build the extractor, extraction judge and connection judge agents."""
    linking_prompt = render_prompt("judge_linking_system.md").strip()
    return (
        Agent(
            _model(extractor_name),
            output_type=PromptedOutput(ExtractionBatch),
            system_prompt=render_prompt("extractor_system.md"),
            retries=5,
        ),
        Agent(
            _model(judge_name),
            output_type=PromptedOutput(ExtractionVerdict),
            system_prompt=render_prompt("judge_system.md"),
            retries=5,
        ),
        Agent(
            _model(judge_name),
            output_type=PromptedOutput(LinkVerdict),
            system_prompt=linking_prompt or render_prompt("judge_system.md"),
            retries=5,
        ),
    )


def run[T: BaseModel](agent: Agent[None, T], prompt: str) -> T:
    """Run an agent once, returning its validated output."""
    return asyncio.run(agent.run(prompt)).output


def list_loaded_models() -> list[str]:
    """List model ids currently loaded in LM Studio."""
    base = global_config.lm_studio_api_base_url
    base = base if "://" in base else f"http://{base}"
    response = httpx.get(f"{base}/v1/models", timeout=10)
    response.raise_for_status()
    return [m["id"] for m in response.json().get("data", [])]
