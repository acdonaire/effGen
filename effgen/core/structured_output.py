"""
Structured output support for the effGen framework.

This module allows agents to return guaranteed-valid structured data
(JSON, YAML, CSV) matching a user-provided schema.

Approaches (in priority order):
1. Grammar-constrained decoding via ``outlines`` (optional, Transformers only)
2. API-level ``response_format`` for API models (OpenAI, Gemini)
3. Post-validation with ``jsonschema`` + retry (universal fallback)

Usage::

    result = agent.run(
        "Extract entities from this text",
        output_schema={
            "type": "object",
            "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
            "required": ["entities"]
        }
    )
    data = json.loads(result.output)  # guaranteed valid

Pydantic support::

    class Entities(BaseModel):
        items: list[str]
    result = agent.run("...", output_model=Entities)
    parsed = result.metadata["parsed"]  # Entities instance
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StructuredOutputConfig:
    """Configuration for structured output constraints.

    Attributes:
        schema: JSON Schema dict that the output must conform to.
        format: Output format — "json" (default), "yaml", or "csv".
        max_retries: Maximum retries on validation failure.
        use_grammar: Whether to attempt grammar-constrained decoding
            (requires ``outlines`` library, Transformers only).
    """
    schema: dict[str, Any] | None = None
    format: str = "json"
    max_retries: int = 3
    use_grammar: bool = True


def validate_json_schema(data: Any, schema: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate data against a JSON Schema.

    Tries ``jsonschema`` library first, falls back to basic structural checks.

    Args:
        data: Parsed JSON data to validate.
        schema: JSON Schema dict.

    Returns:
        (is_valid, error_message)
    """
    try:
        import jsonschema
        jsonschema.validate(data, schema)
        return True, None
    except ImportError:
        logger.debug("jsonschema not installed, using basic validation")
        return _basic_validate(data, schema)
    except Exception as e:
        return False, str(e)


def _basic_validate(data: Any, schema: dict[str, Any]) -> tuple[bool, str | None]:
    """Basic JSON Schema validation without jsonschema library.

    Only checks type, required fields, and properties — not full spec.
    """
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            return False, f"Expected object, got {type(data).__name__}"
        required = schema.get("required", [])
        for key in required:
            if key not in data:
                return False, f"Missing required field: {key}"
        properties = schema.get("properties", {})
        for key, prop_schema in properties.items():
            if key in data:
                valid, err = _basic_validate(data[key], prop_schema)
                if not valid:
                    return False, f"Field '{key}': {err}"
        return True, None

    elif schema_type == "array":
        if not isinstance(data, list):
            return False, f"Expected array, got {type(data).__name__}"
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(data):
                valid, err = _basic_validate(item, items_schema)
                if not valid:
                    return False, f"Item [{i}]: {err}"
        return True, None

    elif schema_type == "string":
        return (True, None) if isinstance(data, str) else (False, f"Expected string, got {type(data).__name__}")
    elif schema_type == "integer":
        return (True, None) if isinstance(data, int) and not isinstance(data, bool) else (False, "Expected integer")
    elif schema_type == "number":
        return (True, None) if isinstance(data, int | float) and not isinstance(data, bool) else (False, "Expected number")
    elif schema_type == "boolean":
        return (True, None) if isinstance(data, bool) else (False, "Expected boolean")
    elif schema_type == "null":
        return (True, None) if data is None else (False, "Expected null")

    # No type constraint or unknown type — accept
    return True, None


def extract_json_from_text(text: str) -> str | None:
    """Extract the first JSON object or array from free text.

    Handles:
    - Markdown code fences (```json ... ```)
    - JSON embedded in prose
    - Trailing commas and unquoted keys (basic cleanup)

    Returns:
        Extracted JSON string, or None if not found.
    """
    # Try markdown code fence first
    fence_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if fence_match:
        return _clean_json(fence_match.group(1).strip())

    # Try to find JSON object or array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        # Find matching end bracket (accounting for nesting)
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == '\\' and in_string:
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    return _clean_json(text[start:i + 1])
    return None


def _clean_json(text: str) -> str:
    """Clean common JSON issues from SLM output."""
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    # Quote unquoted keys
    text = re.sub(r'(?<=[{,])\s*([a-zA-Z_]\w*)\s*:', r' "\1":', text)
    return text


def constrain_output(
    model: Any,
    prompt: str,
    schema: dict[str, Any],
    config: StructuredOutputConfig | None = None,
    generation_kwargs: dict[str, Any] | None = None,
) -> tuple[str, Any]:
    """Generate structured output from a model, constrained to a JSON Schema.

    Attempts strategies in order:
    1. Grammar-constrained decoding (outlines, Transformers only)
    2. API response_format (OpenAI, Gemini)
    3. Retry with validation (universal fallback)

    Args:
        model: The model instance (BaseModel subclass).
        prompt: The user prompt.
        schema: JSON Schema the output must conform to.
        config: Optional StructuredOutputConfig.
        generation_kwargs: Extra kwargs passed to model.generate().

    Returns:
        (json_string, parsed_data) — guaranteed to validate against schema.

    Raises:
        ValueError: If output cannot be constrained after all retries.
    """
    if config is None:
        config = StructuredOutputConfig(schema=schema)

    gen_kwargs = generation_kwargs or {}

    # --- Strategy 1: Grammar-constrained decoding (outlines) ---
    if config.use_grammar and _is_transformers_model(model):
        result = _try_grammar_constrained(model, prompt, schema, gen_kwargs)
        if result is not None:
            return result

    # --- Strategy 2: API response_format ---
    if _is_api_model(model):
        result = _try_api_json_mode(model, prompt, schema, gen_kwargs)
        if result is not None:
            return result

    # --- Strategy 3: Retry with validation (universal fallback) ---
    return _retry_with_validation(model, prompt, schema, config.max_retries, gen_kwargs)


def _is_transformers_model(model: Any) -> bool:
    """Check if model is a local Transformers engine."""
    return type(model).__name__ in ("TransformersEngine",)


def _is_api_model(model: Any) -> bool:
    """Check if model is an API adapter."""
    return type(model).__name__ in ("OpenAIAdapter", "AnthropicAdapter", "GeminiAdapter")


def _try_grammar_constrained(
    model: Any,
    prompt: str,
    schema: dict[str, Any],
    gen_kwargs: dict[str, Any],
) -> tuple[str, Any] | None:
    """Try grammar-constrained decoding via outlines library."""
    try:
        import outlines  # noqa: F401
        from outlines import generate
        from outlines import models as outlines_models

        logger.info("Using outlines grammar-constrained decoding for structured output")
        # outlines requires access to the underlying HF model
        if hasattr(model, 'model') and hasattr(model, 'tokenizer'):
            hf_model = outlines_models.Transformers(model.model, model.tokenizer)
            schema_str = json.dumps(schema)
            generator = generate.json(hf_model, schema_str)
            result = generator(prompt)
            # result is already a parsed object
            json_str = json.dumps(result)
            return json_str, result
    except ImportError:
        logger.debug("outlines not installed, skipping grammar-constrained decoding")
    except Exception as e:
        logger.warning(f"Grammar-constrained decoding failed: {e}")
    return None


def _try_api_json_mode(
    model: Any,
    prompt: str,
    schema: dict[str, Any],
    gen_kwargs: dict[str, Any],
) -> tuple[str, Any] | None:
    """Try API-level JSON mode for API models."""
    try:
        model_type = type(model).__name__
        from ..models.base import GenerationConfig

        config = GenerationConfig(
            temperature=gen_kwargs.get("temperature", 0.3),
            max_tokens=gen_kwargs.get("max_tokens", 2048),
        )

        # Enhance prompt with schema instruction
        enhanced_prompt = (
            f"{prompt}\n\n"
            f"You MUST respond with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            f"Respond ONLY with the JSON object, no other text."
        )

        if model_type == "OpenAIAdapter" and hasattr(model, 'client'):
            # OpenAI supports response_format
            messages = [{"role": "user", "content": enhanced_prompt}]
            response = model.client.chat.completions.create(
                model=model.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            json_str = response.choices[0].message.content
            parsed = json.loads(json_str)
            valid, err = validate_json_schema(parsed, schema)
            if valid:
                return json_str, parsed
            logger.warning(f"API JSON mode output didn't match schema: {err}")

        elif model_type == "GeminiAdapter" and hasattr(model, 'generative_model'):
            # Gemini supports response_mime_type
            result = model.generate(enhanced_prompt, config=config)
            json_str = extract_json_from_text(result.text)
            if json_str:
                parsed = json.loads(json_str)
                valid, err = validate_json_schema(parsed, schema)
                if valid:
                    return json_str, parsed

        else:
            # Anthropic or other — use enhanced prompt
            result = model.generate(enhanced_prompt, config=config)
            json_str = extract_json_from_text(result.text)
            if json_str:
                parsed = json.loads(json_str)
                valid, err = validate_json_schema(parsed, schema)
                if valid:
                    return json_str, parsed

    except Exception as e:
        logger.warning(f"API JSON mode failed: {e}")
    return None


def _retry_with_validation(
    model: Any,
    prompt: str,
    schema: dict[str, Any],
    max_retries: int,
    gen_kwargs: dict[str, Any],
) -> tuple[str, Any]:
    """Generate output and retry until it validates against schema.

    This is the universal fallback that works with any model.
    """
    from ..models.base import GenerationConfig

    errors: list[str] = []

    for attempt in range(max_retries):
        try:
            # Build prompt with increasing urgency about JSON format
            if attempt == 0:
                enhanced_prompt = (
                    f"{prompt}\n\n"
                    f"Respond with valid JSON matching this schema:\n"
                    f"```json\n{json.dumps(schema, indent=2)}\n```\n"
                    f"Respond ONLY with the JSON object."
                )
            else:
                enhanced_prompt = (
                    f"{prompt}\n\n"
                    f"IMPORTANT: Your previous response was not valid JSON. "
                    f"Previous error: {errors[-1]}\n"
                    f"You MUST respond with ONLY a valid JSON object matching:\n"
                    f"```json\n{json.dumps(schema, indent=2)}\n```\n"
                    f"No explanation, no markdown, ONLY the JSON."
                )

            config = GenerationConfig(
                temperature=max(0.1, gen_kwargs.get("temperature", 0.3) - (attempt * 0.1)),
                max_tokens=gen_kwargs.get("max_tokens", 2048),
            )

            result = model.generate(enhanced_prompt, config=config)
            text = result.text if hasattr(result, 'text') else str(result)

            # Extract JSON
            json_str = extract_json_from_text(text)
            if json_str is None:
                errors.append("No JSON object found in response")
                logger.debug(f"Attempt {attempt + 1}/{max_retries}: no JSON found")
                continue

            parsed = json.loads(json_str)
            valid, err = validate_json_schema(parsed, schema)
            if valid:
                logger.info(f"Structured output validated on attempt {attempt + 1}")
                return json_str, parsed

            errors.append(f"Schema validation failed: {err}")
            logger.debug(f"Attempt {attempt + 1}/{max_retries}: {err}")

        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {e}")
            logger.debug(f"Attempt {attempt + 1}/{max_retries}: JSON decode error: {e}")
        except Exception as e:
            errors.append(str(e))
            logger.warning(f"Attempt {attempt + 1}/{max_retries}: unexpected error: {e}")

    raise ValueError(
        f"Failed to generate valid structured output after {max_retries} attempts. "
        f"Errors: {errors}"
    )


def pydantic_model_to_schema(model_class: Any) -> dict[str, Any]:
    """Convert a Pydantic model class to a JSON Schema dict.

    Strips Pydantic-specific ``title`` fields that confuse SLMs into
    echoing the schema instead of producing data.

    Args:
        model_class: A Pydantic BaseModel subclass.

    Returns:
        JSON Schema dict.

    Raises:
        TypeError: If the class is not a Pydantic model.
    """
    if hasattr(model_class, 'model_json_schema'):
        # Pydantic v2
        schema = model_class.model_json_schema()
    elif hasattr(model_class, 'schema'):
        # Pydantic v1
        schema = model_class.schema()
    else:
        raise TypeError(
            f"{model_class} is not a Pydantic model. "
            "It must have model_json_schema() (v2) or schema() (v1)."
        )
    _strip_title_fields(schema)
    return schema


def _strip_title_fields(schema: dict[str, Any]) -> None:
    """Remove ``title`` fields from a JSON Schema in-place.

    Pydantic v2 adds ``title`` to every property and the root schema.
    SLMs often echo these titles back as data, so we strip them.
    """
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        if isinstance(prop, dict):
            prop.pop("title", None)
            # Recurse into nested objects / array items
            if "properties" in prop:
                _strip_title_fields(prop)
            items = prop.get("items")
            if isinstance(items, dict):
                items.pop("title", None)
                if "properties" in items:
                    _strip_title_fields(items)
