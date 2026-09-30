# Week 1 — LLM Playground

A small app for exploring what happens when an application calls an LLM:
requests and responses, model parameters, message roles, tokens, structured
output, latency, and cost. It works with both **OpenAI** and **Claude**.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your API keys to .env
```

## Experiments (terminal)

Each script maps to a step in the Week 1 lab. Pass `openai` or `claude`:

| Script | Lab step | What to observe |
|---|---|---|
| `experiments/step2_first_call.py` | Step 2 | Fields in the full response object |
| `experiments/step3_max_tokens.py` | Step 3 | The output limit is a ceiling; low limits truncate |
| `experiments/step4_roles.py` | Step 4.2 | System instructions change the answer |
| `experiments/step4_history.py` | Step 4.3 | The model only "remembers" history you resend |
| `experiments/step5_tokens.py` | Step 5 | Tokens ≠ words |
| `experiments/step6_structured.py` | Step 6 | Valid JSON ≠ correct information |
| `experiments/step7_latency_cost.py` | Step 7 | Latency and cost across providers |

```bash
python experiments/step2_first_call.py claude
```

## Web app

```bash
streamlit run app.py
```

## Architecture

```
User ──► Streamlit UI (app.py)
             │  provider, model, instructions, prompt, schema
             ▼
         llm_client.call_llm()  ── unified interface
             │
     ┌───────┴────────┐
     ▼                ▼
 OpenAI            Anthropic
 Responses API     Messages API
     │                │
     └───────┬────────┘
             ▼
         LLMResult: text / parsed JSON / tokens / latency / cost
```
