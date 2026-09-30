"""Step 4.3 — With vs. without conversation history.

The model does not remember earlier API calls. The only reason it "knows" your
background in Experiment 1 is that we send the earlier messages again.
"""

from _common import banner, get_provider, print_usage
from llm_client import call_llm

provider = get_provider()
system = "You are a career advisor. Provide concise, practical advice."
current_question = "Enterprise AI products. What skills should I develop?"

history = [
    {"role": "user", "content": "I am a software engineer interested in transitioning into AI product management."},
    {"role": "assistant", "content": "Your engineering background could help. What type of AI products interest you?"},
    {"role": "user", "content": current_question},
]

with_history = call_llm(provider, system=system, messages=history,
                        max_output_tokens=1500, effort="low")
banner("EXPERIMENT 1: WITH CONVERSATION HISTORY")
print(with_history.text)
print()
print_usage(with_history)

without_history = call_llm(provider, system=system, messages=current_question,
                           max_output_tokens=1500, effort="low")
banner("EXPERIMENT 2: WITHOUT CONVERSATION HISTORY")
print(without_history.text)
print()
print_usage(without_history)

# Compare: does Experiment 2 still mention the engineering background?
# Compare input tokens: history costs tokens on every single call.
