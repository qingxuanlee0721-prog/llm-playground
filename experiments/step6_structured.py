"""Step 6 — Coffee Chat Intent Analyzer: natural language → validated JSON."""

from typing import Literal

from pydantic import BaseModel

from _common import banner, get_provider, print_usage
from llm_client import call_llm


class CoffeeChatIntent(BaseModel):
    intent: Literal["career_advice", "networking", "skill_learning", "other"]
    topic: str
    looking_for: str
    is_asking_for_connection: bool


provider = get_provider()
instructions = (
    "Analyze Coffee Chat requests. Extract the user's intent and requirements. "
    "Do not invent information."
)

test_inputs = [
    # Clear career-advice request
    "I'm a software engineer transitioning into AI product management. I'd love to meet "
    "someone who has already made this transition and learn about their experience.",
    # Networking
    "Anyone in the Bay Area working on developer tools? Would be great to grab coffee.",
    # Skill learning
    "Can someone walk me through how RAG pipelines are evaluated? I want to learn the basics.",
    # Ambiguous — watch whether the model invents details
    "Hi!",
]

for text in test_inputs:
    result = call_llm(provider, system=instructions, messages=text,
                      schema=CoffeeChatIntent, max_output_tokens=2000, effort="low")
    banner(f"INPUT: {text}")
    if result.parsed is None:
        print(f"No parsed result ({result.status}).")
    else:
        print(result.parsed.model_dump_json(indent=2))
    print_usage(result)

# Discussion: for "Hi!", the schema forces every field to be filled. Did the model
# make something up? How would you change the schema to allow "unknown"?
