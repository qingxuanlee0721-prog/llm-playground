"""Step 5 — Measure token usage for short, medium, and long answers."""

from _common import banner, get_provider, print_usage
from llm_client import call_llm

provider = get_provider()
prompts = [
    "What is an LLM?",
    "Explain what an LLM is in 200 words.",
    "Explain what an LLM is in 1,000 words, including examples.",
]

rows = []
for i, prompt in enumerate(prompts, start=1):
    result = call_llm(provider, messages=prompt, max_output_tokens=4000, effort="low")
    banner(f"TEST {chr(64 + i)}: {prompt}")
    print(result.text[:500] + ("..." if len(result.text) > 500 else ""))
    print()
    print_usage(result)
    rows.append((prompt, result.input_tokens, result.output_tokens, len(result.text.split())))

banner("RECORD THESE IN YOUR NOTES")
print(f"{'Prompt':<60} {'In':>6} {'Out':>6} {'Words':>6}")
for prompt, tokens_in, tokens_out, words in rows:
    print(f"{prompt[:58]:<60} {tokens_in:>6} {tokens_out:>6} {words:>6}")
# Notice: output tokens ≠ words (and thinking tokens are counted as output too).
