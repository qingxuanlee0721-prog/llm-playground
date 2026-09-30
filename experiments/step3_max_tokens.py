"""Step 3 — The output-token limit is a ceiling, not a target."""

from _common import banner, get_provider, print_usage
from llm_client import call_llm

provider = get_provider()
prompt = "Explain the difference between an LLM and an AI agent in detail, including examples."

for limit in [100, 300, 800]:
    result = call_llm(provider, messages=prompt, max_output_tokens=limit, effort="low")
    banner(f"MAX OUTPUT TOKENS: {limit}")
    print_usage(result)
    print("\nModel response:")
    print(result.text or "[No visible text returned]")

# Things to notice:
# 1. With a low limit the status becomes "incomplete" — the answer is cut off.
# 2. Reasoning/thinking tokens also count against this limit, so a tiny limit can
#    return no visible text at all.
# 3. Try asking "answer in under 60 words" instead: the model plans a short answer,
#    which is different from cutting a long answer off.
