from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


Provider = Literal["openai", "google", "huggingface", "api", "finetune"]
Mode = Literal["chat", "completion"]
ActionSet = Literal["playwright", "id_accessibility_tree", "som", "webrl_id"]


class ModelConfig(BaseModel):
    model_config = {
        "protected_namespaces": ()
    }
    provider: Provider
    model: str
    mode: Mode
    model_endpoint: Optional[str] = None
    # Optional per-model generation config (overrides CLI defaults)
    class GenConfig(BaseModel):
        model_config = {
            "protected_namespaces": ()
        }
        temperature: Optional[float] = Field(default=None, ge=0.0)
        top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
        context_length: Optional[int] = Field(default=None, gt=0)
        max_tokens: Optional[int] = Field(default=None, gt=0)
        stop_token: Optional[str] = None
        max_obs_length: Optional[int] = Field(default=None, gt=0)
        max_retry: Optional[int] = Field(default=None, ge=0)

    gen: Optional[GenConfig] = None


class MetaDataConfig(BaseModel):
    model_config = {
        "protected_namespaces": ()
    }
    prompt_constructor: str
    task_name: str
    description: str
    answer_phrase: str
    keywords: List[str]
    action_splitter: str


class InstructionConfig(BaseModel):
    model_config = {
        "protected_namespaces": ()
    }
    # Prompt basics
    intro: str
    examples: List[List[str]]
    template: str
    meta_data: MetaDataConfig

    # Reward-guided root-level params (optional; CLI may override)
    num_samples: Optional[int] = Field(default=None, gt=0)
    temperature: Optional[float] = Field(default=None, ge=0.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    max_refinements: Optional[int] = Field(default=None, ge=0)

    # Default action set for agent (optional)
    action_set_tag: Optional[ActionSet] = None

    # Policy and reward model configs (optional; CLI may override)
    policy_model: Optional[ModelConfig] = None
    reward_model: Optional[ModelConfig] = None

    # Optional runtime block for runner scripts
    class RuntimeConfig(BaseModel):
        model_config = {
            "protected_namespaces": ()
        }
        real_env: Optional[bool] = False
        render: Optional[bool] = False
        observation_type: Optional[Literal[
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "html",
            "image",
            "image_som",
            "webrl",
        ]] = None
        viewport_width: Optional[int] = Field(default=None, gt=0)
        viewport_height: Optional[int] = Field(default=None, gt=0)
        sleep_after_execution: Optional[float] = Field(default=None, ge=0.0)
        max_steps: Optional[int] = Field(default=None, gt=0)
        planner_ip: Optional[str] = None
        output_response: Optional[bool] = None
        log_level: Optional[Literal["DEBUG", "INFO", "WARNING", "ERROR"]] = None

    runtime: Optional[RuntimeConfig] = None

    @field_validator("examples")
    def _validate_examples(cls, v: List[List[str]]) -> List[List[str]]:
        for ex in v:
            if not isinstance(ex, list) or len(ex) not in (2, 3):
                raise ValueError("each example must be a list of 2 (text) or 3 (text with image path) strings")
            if not all(isinstance(x, str) for x in ex):
                raise ValueError("all example entries must be strings")
        return v


def load_and_validate_instruction(path: str) -> InstructionConfig:
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return InstructionConfig(**data)


