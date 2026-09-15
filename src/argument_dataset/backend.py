"""
LM Studio agents built with pydantic-ai.

Every request is standalone (no conversation memory).
"""

import asyncio
from pathlib import Path
from string import Template

import httpx
from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from src.argument_dataset.output_models import (
    ExtractionBatch,
    ExtractionVerdict,
    FrameClassification,
    GeneratedPremises,
    LinkVerdict,
    PremiseSupportVerdict,
    StyledDocument,
)
from src.configuration import RoleModelSettings
from src.configuration import config as global_config
from src.data_models.data_models import InterpretativeFrame

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


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


def _settings(role: RoleModelSettings) -> OpenAIChatModelSettings:
    """
    Build the sampling settings of one agent role.

    Passing these explicitly stops every request from inheriting whatever
    defaults the model happened to be loaded with in LM Studio.

    ``reasoning_effort="none"`` is the one that matters for wall-clock: on a
    thinking model the reasoning block is most of the output and none of the
    answer. LM Studio only honours it as a top-level field - the same key
    nested in ``chat_template_kwargs`` is silently ignored.
    """
    settings = OpenAIChatModelSettings(
        temperature=role.temperature,
        max_tokens=role.max_tokens or global_config.llm.max_tokens,
        timeout=global_config.llm.timeout,
    )
    if role.top_p is not None:
        settings["top_p"] = role.top_p
    if role.seed is not None:
        settings["seed"] = role.seed
    if role.reasoning_effort is not None:
        settings["openai_reasoning_effort"] = role.reasoning_effort
    return settings


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
            model_settings=_settings(global_config.llm.extractor),
            retries=3,
        ),
        Agent(
            _model(judge_name),
            output_type=PromptedOutput(ExtractionVerdict),
            system_prompt=render_prompt("judge_system.md"),
            model_settings=_settings(global_config.llm.judge),
            retries=3,
        ),
        Agent(
            _model(judge_name),
            output_type=PromptedOutput(LinkVerdict),
            system_prompt=linking_prompt or render_prompt("judge_system.md"),
            model_settings=_settings(global_config.llm.judge),
            retries=3,
        ),
    )


def make_frame_classifier(model_name: str) -> Agent[None, FrameClassification]:
    """Build the agent assigning an InterpretativeFrame to a single argument."""
    return Agent(
        _model(model_name),
        output_type=PromptedOutput(FrameClassification),
        system_prompt=render_prompt("frame_classifier_system.md"),
        model_settings=_settings(global_config.llm.extractor),
        retries=3,
    )


def make_premise_generator(model_name: str) -> Agent[None, GeneratedPremises]:
    """Build the agent proposing supporting premises for a bare conclusion."""
    return Agent(
        _model(model_name),
        output_type=PromptedOutput(GeneratedPremises),
        system_prompt=render_prompt("premise_generator_system.md"),
        model_settings=_settings(global_config.llm.generator),
        retries=3,
    )


def make_premise_judge(model_name: str) -> Agent[None, PremiseSupportVerdict]:
    """Build the agent scoring how well candidate premises support a claim."""
    return Agent(
        _model(model_name),
        output_type=PromptedOutput(PremiseSupportVerdict),
        system_prompt=render_prompt("premise_judge_system.md"),
        model_settings=_settings(global_config.llm.judge),
        retries=3,
    )


def make_document_styler(model_name: str) -> Agent[None, StyledDocument]:
    """Build the agent styling a carrier document around verbatim sentences."""
    return Agent(
        _model(model_name),
        output_type=PromptedOutput(StyledDocument),
        system_prompt=render_prompt("document_styler_system.md"),
        model_settings=_settings(global_config.llm.styler),
        retries=3,
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
