"""Step 2 — Send your first request and inspect the full response object."""

import json

from _common import banner, get_provider, print_usage
from llm_client import call_llm

provider = get_provider()

result = call_llm(
    provider,
    system="You are a helpful AI assistant.",
    messages="Explain what an LLM is in two sentences.",
)

banner("GENERATED TEXT")
print(result.text)

banner("FULL API RESPONSE")
# Look for: id, model, status / stop_reason, output / content, usage
print(json.dumps(result.raw, indent=2, default=str))

banner("SUMMARY")
print_usage(result)
