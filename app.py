"""
Week 1 — LLM Playground (Streamlit).

Run locally:  streamlit run app.py
"""

import hmac
import keyword
import os

import streamlit as st
from pydantic import create_model

# =========================================
# 1. PAGE CONFIGURATION
# =========================================

st.set_page_config(page_title="LLM Playground", page_icon="🤖", layout="wide")

# On Streamlit Cloud, keys live in st.secrets. Copy them into environment
# variables so the OpenAI / Anthropic SDKs find them the same way as locally.
try:
    for secret_name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "APP_PASSWORD"):
        if secret_name in st.secrets:
            os.environ[secret_name] = st.secrets[secret_name]
except FileNotFoundError:
    pass  # No secrets file locally — .env is used instead.

from llm_client import MODEL_CHOICES, PRICES, PROVIDERS, call_llm  # noqa: E402

# =========================================
# 2. PASSWORD GATE
# =========================================
# A public app with your API key means strangers can spend your credits.
# If APP_PASSWORD is set, visitors must enter it first.

app_password = os.getenv("APP_PASSWORD")
if app_password and not st.session_state.get("authenticated"):
    st.title("🤖 LLM Playground")
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered, app_password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()

st.title("🤖 LLM Playground")
st.caption("Week 1 — Explore LLM API requests, structured output, token usage, latency, and cost.")

available = [
    p for p, key in (("openai", "OPENAI_API_KEY"), ("claude", "ANTHROPIC_API_KEY"))
    if os.getenv(key)
]
if not available:
    st.error("No API key found. Add OPENAI_API_KEY and/or ANTHROPIC_API_KEY to .env (or Streamlit secrets).")
    st.stop()

# =========================================
# 3. DYNAMIC JSON OUTPUT MODEL
# =========================================

SUPPORTED_TYPES = {"str": str, "int": int, "float": float, "bool": bool, "list[str]": list[str]}


def build_output_model(schema_text: str):
    """Turn lines like `intent: str` into a Pydantic model."""
    fields = {}
    for line in schema_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Invalid field definition: {line}")
        name, type_name = (part.strip() for part in line.split(":", 1))
        if not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_"):
            raise ValueError(f"Invalid field name: {name}")
        if type_name not in SUPPORTED_TYPES:
            raise ValueError(f"Unsupported type: {type_name}")
        if name in fields:
            raise ValueError(f"Duplicate field: {name}")
        fields[name] = (SUPPORTED_TYPES[type_name], ...)
    if not fields:
        raise ValueError("Define at least one JSON field.")
    return create_model("CustomOutput", **fields)


# =========================================
# 4. SESSION STATE
# =========================================
# Streamlit re-runs this whole script on every click. Anything we want to keep
# between runs must live in st.session_state — just like an LLM app must keep
# conversation state itself, because the model is stateless.

if "last_result" not in st.session_state:
    st.session_state.last_result = None

left_col, right_col = st.columns([1, 1], gap="large")

# =========================================
# LEFT COLUMN: CONFIGURATION & INPUT
# =========================================

with left_col:
    st.subheader("Configuration & Input")

    provider_col, model_col = st.columns(2)
    with provider_col:
        provider = st.radio(
            "Provider", [p for p in PROVIDERS if p in available],
            horizontal=True, format_func=lambda p: {"openai": "OpenAI", "claude": "Claude"}[p],
        )
    with model_col:
        model = st.selectbox("Model", MODEL_CHOICES[provider], accept_new_options=True)

    token_col, effort_col = st.columns(2)
    with token_col:
        max_output_tokens = st.number_input(
            "Max Output Tokens", min_value=16, max_value=8000, value=1500, step=100,
            help="Capped at 8,000 so a public demo can't run up a large bill.",
        )
    with effort_col:
        effort = st.selectbox("Reasoning effort", ["low", "medium", "high"], index=0)

    instructions = st.text_area("System Instructions", value="You are a helpful AI assistant.", height=68)
    prompt = st.text_area(
        "User Prompt",
        value="I'm a software engineer transitioning into AI product management. What skills should I develop?",
        height=100,
    )

    output_format = st.radio("Output Format", ["Text", "Structured JSON"], horizontal=True)
    schema_text = ""
    if output_format == "Structured JSON":
        schema_text = st.text_area(
            "JSON Fields",
            value="intent: str\ntopic: str\nlooking_for: str\nis_asking_for_connection: bool",
            height=95,
        )
        st.caption("Supported types: " + ", ".join(SUPPORTED_TYPES))

    with st.expander("Cost Settings (USD per 1M tokens)"):
        default_in, default_out = PRICES.get(model, (0.0, 0.0))
        input_price = st.number_input("Input price", min_value=0.0, value=default_in, format="%.4f")
        output_price = st.number_input("Output price", min_value=0.0, value=default_out, format="%.4f")
        st.caption("Defaults are standard uncached prices; verify them on the provider's pricing page.")

    generate = st.button("Generate Response", type="primary", use_container_width=True)

# =========================================
# EXECUTE API REQUEST
# =========================================

if generate:
    if not prompt.strip():
        st.warning("Please enter a user prompt.")
    else:
        try:
            schema = build_output_model(schema_text) if output_format == "Structured JSON" else None
            with st.spinner("Generating response..."):
                result = call_llm(
                    provider,
                    system=instructions,
                    messages=prompt,
                    model=model.strip(),
                    max_output_tokens=int(max_output_tokens),
                    effort=effort,
                    schema=schema,
                    prices=(input_price, output_price),
                )
            st.session_state.last_result = result
        except Exception as error:
            st.error(f"API request failed: {error}")

# =========================================
# RIGHT COLUMN: METRICS & RESPONSE
# =========================================

with right_col:
    result = st.session_state.last_result
    st.subheader("API Metrics")
    m1, m2, m3, m4 = st.columns(4)

    if result is not None:
        m1.metric("Input", result.input_tokens)
        m2.metric("Output", result.output_tokens)
        m3.metric("Latency", f"{result.latency_s:.2f}s")
        m4.metric("Cost", f"${result.cost_usd:.5f}" if result.cost_usd is not None else "—")
        st.caption(
            f"{result.provider} · {result.model} · total tokens: {result.total_tokens} · "
            f"status: {result.status} ({result.stop_reason})"
        )
        if result.status == "incomplete":
            st.warning("Response incomplete: the output-token limit was reached.")
        elif result.status == "refused":
            st.warning("The model declined this request.")
    else:
        for column, label in zip((m1, m2, m3, m4), ("Input", "Output", "Latency", "Cost")):
            column.metric(label, "—")
        st.caption("Metrics will appear after generation.")

    st.subheader("Generated Response")
    with st.container(border=True):
        if result is None:
            st.caption("Your generated response will appear here.")
        elif result.parsed is not None:
            st.json(result.parsed.model_dump())
        elif result.text:
            st.markdown(result.text)
        else:
            st.info("No visible text returned.")

    if result is not None:
        with st.expander("Full API Response"):
            st.json(result.raw)
