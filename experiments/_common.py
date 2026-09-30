"""Shared helpers for the experiment scripts."""

import sys
from pathlib import Path

# Make `llm_client` importable when running `python experiments/xxx.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import PROVIDERS  # noqa: E402


def get_provider() -> str:
    """Read the provider from the command line: `python step2_first_call.py claude`."""
    provider = sys.argv[1] if len(sys.argv) > 1 else "openai"
    if provider not in PROVIDERS:
        sys.exit(f"Provider must be one of {PROVIDERS}")
    return provider


def banner(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def print_usage(result) -> None:
    cost = f"${result.cost_usd:.6f}" if result.cost_usd is not None else "unknown"
    print(f"Status: {result.status} ({result.stop_reason})")
    print(f"Input tokens: {result.input_tokens} | Output tokens: {result.output_tokens}")
    print(f"Latency: {result.latency_s:.2f}s | Estimated cost: {cost}")
