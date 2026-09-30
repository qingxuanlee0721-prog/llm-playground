"""Step 7 — Latency and cost, across both providers.

Run: python step7_latency_cost.py        (compares OpenAI vs Claude)
"""

import sys

from _common import banner
from llm_client import DEFAULT_MODELS, call_llm

tests = [
    ("Short prompt", "openai", "What is an LLM? One sentence."),
    ("Long prompt", "openai", "Explain what an LLM is in 500 words, with examples."),
    ("Different model", "claude", "Explain what an LLM is in 500 words, with examples."),
]

rows = []
for name, provider, prompt in tests:
    try:
        result = call_llm(provider, messages=prompt, max_output_tokens=3000, effort="low")
    except Exception as error:  # e.g. missing key for one provider
        print(f"{name} failed: {error}", file=sys.stderr)
        continue
    rows.append((name, result))

banner("RESULTS")
print(f"{'Test':<16} {'Model':<20} {'In':>6} {'Out':>6} {'Latency':>9} {'Cost':>11}")
for name, r in rows:
    cost = f"${r.cost_usd:.6f}" if r.cost_usd is not None else "unknown"
    print(f"{name:<16} {r.model[:20]:<20} {r.input_tokens:>6} {r.output_tokens:>6} "
          f"{r.latency_s:>8.2f}s {cost:>11}")

banner("AT SCALE")
for name, r in rows:
    if r.cost_usd is not None:
        print(f"{name}: 10,000 requests/day ≈ ${r.cost_usd * 10_000:,.2f}/day")
print(f"\nDefault models: {DEFAULT_MODELS}")
