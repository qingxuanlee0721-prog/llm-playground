"""
A unified LLM client for OpenAI and Claude.

The rest of the app calls `call_llm(...)` and always gets back the same
`LLMResult` shape, no matter which provider served the request. This is the
file Week 2 will reuse as `core/llm_client.py`.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Type

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

PROVIDERS = ["openai", "claude"]

DEFAULT_MODELS = {
    "openai": "gpt-5.4-mini",
    "claude": "claude-opus-5-5",
}

# Models offered in the UI dropdown. You can also type any other model ID.
MODEL_CHOICES = {
    "openai": ["gpt-5.4-mini"],
    "claude": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"],
}

# Standard (uncached) prices in USD per 1M tokens: (input, output).
# Prices change — check the provider pricing pages before relying on these.
PRICES = {
    "gpt-5.4-mini": (0.75, 4.50),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


@dataclass
class LLMResult:
    provider: str
    model: str
    text: str
    parsed: Optional[BaseModel]
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: Optional[float]
    status: str          # "completed", "incomplete", or "refused"
    stop_reason: str     # provider-specific reason, e.g. "max_tokens"
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def estimate_cost(model: str, input_tokens: int, output_tokens: int,
                  prices: Optional[tuple] = None) -> Optional[float]:
    """cost = input_tokens × input_price/1M + output_tokens × output_price/1M"""
    prices = prices or PRICES.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def call_llm(
    provider: str,
    messages: list | str,
    system: str = "",
    model: Optional[str] = None,
    max_output_tokens: int = 2000,
    effort: Optional[str] = None,
    schema: Optional[Type[BaseModel]] = None,
    prices: Optional[tuple] = None,
) -> LLMResult:
    """
    Send one request to the chosen provider.

    messages: a plain string, or a list like
              [{"role": "user", "content": "..."}, {"role": "assistant", ...}]
    effort:   "low" / "medium" / "high" reasoning effort, or None for the default
    schema:   a Pydantic model; when given, the reply is structured JSON
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    model = model or DEFAULT_MODELS[provider]

    start = time.perf_counter()
    if provider == "openai":
        result = _call_openai(model, messages, system, max_output_tokens, effort, schema)
    else:
        result = _call_claude(model, messages, system, max_output_tokens, effort, schema)
    result["latency_s"] = time.perf_counter() - start

    # Price by the model that actually served the request (a Claude fallback may
    # differ). OpenAI returns a dated snapshot name, so fall back to the requested ID.
    priced_model = result["model"] if result["model"] in PRICES else model
    result["cost_usd"] = estimate_cost(
        priced_model, result["input_tokens"], result["output_tokens"], prices
    )
    return LLMResult(provider=provider, **result)


# ---------------------------------------------------------------------------
# OpenAI — Responses API
# ---------------------------------------------------------------------------

def _call_openai(model, messages, system, max_output_tokens, effort, schema):
    from openai import OpenAI

    client = OpenAI()
    kwargs = dict(
        model=model,
        instructions=system or None,
        input=messages,
        max_output_tokens=max_output_tokens,
    )
    if effort:
        kwargs["reasoning"] = {"effort": effort}

    if schema is not None:
        response = client.responses.parse(text_format=schema, **kwargs)
        parsed = response.output_parsed
    else:
        response = client.responses.create(**kwargs)
        parsed = None

    incomplete = response.incomplete_details
    return dict(
        model=response.model,
        text=response.output_text or "",
        parsed=parsed,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        status="completed" if response.status == "completed" else "incomplete",
        stop_reason=incomplete.reason if incomplete else response.status,
        raw=response.model_dump(),
    )


# ---------------------------------------------------------------------------
# Claude — Messages API
# ---------------------------------------------------------------------------

def _strict_json_schema(schema: Type[BaseModel]) -> dict:
    """Pydantic schema → the strict JSON schema Claude's structured output expects."""
    json_schema = schema.model_json_schema()

    def close_objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
            for value in node.values():
                close_objects(value)
        elif isinstance(node, list):
            for item in node:
                close_objects(item)

    close_objects(json_schema)
    return json_schema


def _call_claude(model, messages, system, max_output_tokens, effort, schema):
    import anthropic

    client = anthropic.Anthropic()
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    output_config: dict[str, Any] = {}
    if effort:
        output_config["effort"] = effort
    if schema is not None:
        output_config["format"] = {
            "type": "json_schema",
            "schema": _strict_json_schema(schema),
        }

    kwargs = dict(
        model=model,
        max_tokens=max_output_tokens,
        messages=messages,
        # If a safety classifier declines, retry server-side on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if system:
        kwargs["system"] = system
    if output_config:
        kwargs["output_config"] = output_config

    response = client.beta.messages.create(**kwargs)

    # response.content is a list of blocks (thinking, text, ...). Keep the text.
    text = "".join(block.text for block in response.content if block.type == "text")

    if response.stop_reason == "refusal":
        status = "refused"
    elif response.stop_reason == "max_tokens":
        status = "incomplete"
    else:
        status = "completed"

    parsed = None
    if schema is not None and status == "completed":
        parsed = schema.model_validate_json(text)

    return dict(
        model=response.model,
        text=text,
        parsed=parsed,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        status=status,
        stop_reason=response.stop_reason or "",
        raw=response.model_dump(),
    )
