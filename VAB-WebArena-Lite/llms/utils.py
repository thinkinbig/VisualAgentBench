import argparse
import logging
import time
from typing import Any, Optional
from .json_validator import JSONResponseValidator
from .types import LLMResponse

try:
    from vertexai.preview.generative_models import Image
    from llms import generate_from_gemini_completion
except:
    print('Google Cloud not set up, skipping import of vertexai.preview.generative_models.Image and llms.generate_from_gemini_completion')

from llms import (
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    generate_with_api,
    lm_config,
)

APIInput = str | list[Any] | dict[str, Any]


def build_api_input_for_text(
    cfg: lm_config.LMConfig,
    system_text: str,
    user_text: str,
) -> APIInput:
    """Build provider/mode-specific API input for a simple system+user text pair.
    Keeps provider branching out of agent code while delegating sending to call_llm.
    """
    if cfg.provider == "openai":
        if cfg.mode == "chat":
            return [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
        elif cfg.mode == "completion":
            return f"{system_text}\n\n{user_text}"
        else:
            raise ValueError(f"OpenAI models do not support mode {cfg.mode}")
    elif cfg.provider == "google":
        if cfg.mode == "completion":
            return [system_text, user_text]
        else:
            raise ValueError(f"Gemini models do not support mode {cfg.mode}")
    elif cfg.provider == "huggingface":
        return f"{system_text}\n\n{user_text}"
    elif cfg.provider in ["api", "finetune"]:
        if cfg.mode == "chat":
            return [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
        else:
            return f"{system_text}\n\n{user_text}"
    else:
        raise NotImplementedError(f"Provider {cfg.provider} not implemented")

def call_llm(
    lm_config: lm_config.LMConfig,
    prompt: APIInput,
    api_key = None,
    base_url = None
) -> str:
    """Unified LLM call with up to 3 retries across providers.

    Retries on exceptions or empty-string responses with small backoff.
    """
    max_attempts = 3
    backoff_seconds = 1.0
    last_err: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response: str
            if lm_config.provider == "openai":
                if lm_config.mode == "chat":
                    assert isinstance(prompt, list)
                    # Use safe defaults if not provided in gen_config
                    context_len = lm_config.gen_config.get("context_length", 4096)
                    max_tokens = lm_config.gen_config.get("max_tokens", 512)
                    temperature = lm_config.gen_config.get("temperature", 1.0)
                    top_p = lm_config.gen_config.get("top_p", 0.9)
                    response = generate_from_openai_chat_completion(
                        messages=prompt,
                        model=lm_config.model,
                        temperature=temperature,
                        top_p=top_p,
                        context_length=context_len,
                        max_tokens=max_tokens,
                        stop_token=None,
                    )
                elif lm_config.mode == "completion":
                    assert isinstance(prompt, str)
                    response = generate_from_openai_completion(
                        prompt=prompt,
                        model=lm_config.model,
                        temperature=lm_config.gen_config["temperature"],
                        max_tokens=lm_config.gen_config["max_tokens"],
                        top_p=lm_config.gen_config["top_p"],
                        stop_token=lm_config.gen_config["stop_token"],
                        api_key=api_key,
                        base_url=base_url
                    )
                else:
                    raise ValueError(
                        f"OpenAI models do not support mode {lm_config.mode}"
                    )
            elif lm_config.provider == "huggingface":
                assert isinstance(prompt, str)
                response = generate_from_huggingface_completion(
                    prompt=prompt,
                    model_endpoint=lm_config.gen_config["model_endpoint"],
                    temperature=lm_config.gen_config["temperature"],
                    top_p=lm_config.gen_config["top_p"],
                    stop_sequences=lm_config.gen_config["stop_sequences"],
                    max_new_tokens=lm_config.gen_config["max_new_tokens"],
                )
            elif lm_config.provider == "google":
                assert isinstance(prompt, list)
                assert all(
                    [isinstance(p, str) or isinstance(p, Image) for p in prompt]
                )
                response = generate_from_gemini_completion(
                    prompt=prompt,
                    engine=lm_config.model,
                    temperature=lm_config.gen_config["temperature"],
                    max_tokens=lm_config.gen_config["max_tokens"],
                    top_p=lm_config.gen_config["top_p"],
                )
            elif lm_config.provider in ["api", "finetune"]:
                args = {
                    "temperature": lm_config.gen_config["temperature"],   # openai, gemini, claude
                    "max_tokens": lm_config.gen_config["max_tokens"],     # openai, gemini, claude
                    "top_k": lm_config.gen_config["top_p"],               # qwen
                }
                response = generate_with_api(prompt, lm_config.model, args)
            else:
                raise NotImplementedError(
                    f"Provider {lm_config.provider} not implemented"
                )

            # If we got a non-empty response, return it
            if isinstance(response, str) and response.strip() != "":
                return response

            logging.warning(
                f"LLM returned empty response on attempt {attempt}/{max_attempts}. Retrying..."
            )
        except Exception as e:
            last_err = e
            logging.warning(
                f"LLM call failed on attempt {attempt}/{max_attempts}: {e}"
            )

        # Backoff before next attempt if any left
        if attempt < max_attempts:
            time.sleep(backoff_seconds)
            backoff_seconds *= 1.5

    # If all attempts failed, return empty string (callers handle fallbacks)
    if last_err is not None:
        logging.error(f"LLM call failed after {max_attempts} attempts: {last_err}")
    return ""


def call_llm_with_validation(
    lm_config: lm_config.LMConfig,
    prompt: APIInput,
    api_key = None,
    base_url = None,
    validate_json: bool = True
) -> LLMResponse:
    """Call LLM with JSON response validation and retry logic.
    
    Args:
        lm_config: Language model configuration
        prompt: API input prompt
        api_key: Optional API key override
        base_url: Optional base URL override
        validate_json: Whether to validate and fix JSON responses
        
    Returns:
        LLMResponse object with validation results
    """
    validator = JSONResponseValidator()
    max_attempts = 3
    
    for attempt in range(max_attempts):
        # Get raw response from LLM
        raw_response = call_llm(lm_config, prompt, api_key, base_url)
        
        if not raw_response.strip():
            continue
            
        if not validate_json:
            # Return without validation
            return LLMResponse(
                raw_response=raw_response,
                is_valid=True
            )
        
        # Validate and potentially fix the response
        validated_response = validator.validate_and_retry(raw_response)
        
        if validated_response.is_valid:
            logging.info(f"Successfully validated LLM response on attempt {attempt + 1}")
            return validated_response
        
        # Log validation errors for debugging
        logging.warning(
            f"Attempt {attempt + 1} validation failed: {validated_response.validation_errors}"
        )
        
        # If this isn't the last attempt, we could modify the prompt to request better formatting
        if attempt < max_attempts - 1:
            logging.info("Retrying with format-focused prompt...")
            # You could modify the prompt here to emphasize JSON formatting
    
    # Return the last attempt even if invalid
    logging.error("All validation attempts failed, returning last response")
    return validated_response if 'validated_response' in locals() else LLMResponse(
        raw_response="",
        is_valid=False,
        validation_errors=["No valid response received"]
    )
