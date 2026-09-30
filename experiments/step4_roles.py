"""Step 4.2 — Same user prompt, different system instructions."""

from _common import banner, get_provider, print_usage
from llm_client import call_llm

provider = get_provider()
user_prompt = "Explain what an LLM is."

instructions_list = [
    "You are a helpful AI assistant.",
    "You are a patient teacher. Explain concepts using simple language and everyday analogies.",
    "You are an ML engineer. Explain concepts using precise technical terminology.",
]

for i, instructions in enumerate(instructions_list, start=1):
    result = call_llm(provider, system=instructions, messages=user_prompt,
                      max_output_tokens=1500, effort="low")
    banner(f"EXPERIMENT {i}: {instructions}")
    print(result.text or "[No visible text returned]")
    print()
    print_usage(result)
