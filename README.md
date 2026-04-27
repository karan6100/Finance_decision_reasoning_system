
# 📊 Finance Decision Assistant 
- A structured finance reasoning system that evaluates user queries through a decision gate to assess intent and risk, then dynamically selects the appropriate strategy (reasoning, retrieval, or refusal). 
- It generates responses via controlled reasoning and validates them for factual accuracy, logical soundness, and safety. 
- Based on validation, the system either delivers a refined answer, retries with improved reasoning, requests specific clarification, or refuses with a clear explanation—ensuring reliable, non-speculative, and context-aware outputs without offering unsafe or unsupported financial advice.

# Docker Space

This project runs:
- Streamlit UI on port `7860`
- FastAPI backend on port `8000` (internal)

## Local run (optional)

```bash
pip install -r requirements.txt
cd src
uvicorn backend_api:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
cd src
streamlit run streamlit_app.py --server.port 7860 --server.address 0.0.0.0
```

## Logging for Debugging and Production

The API now emits structured request and pipeline logs with a request correlation ID.

Environment variables:
- `LOG_LEVEL` (`DEBUG`, `INFO`, `WARNING`, `ERROR`) - default `INFO`
- `LOG_FORMAT` (`text` or `json`) - default `text`

Local debugging (readable logs):

```bash
cd src
LOG_LEVEL=DEBUG LOG_FORMAT=text uvicorn backend_api:app --host 0.0.0.0 --port 8000
```

Production tracing (JSON logs):

```bash
cd src
LOG_LEVEL=INFO LOG_FORMAT=json uvicorn backend_api:app --host 0.0.0.0 --port 8000
```

How to trace one request end-to-end:
1. Send `X-Request-ID` header from client/gateway.
2. The API echoes it back in the response header `X-Request-ID`.
3. Search logs by `request_id` to connect `request.start`, `analyze.*`, `pipeline.*`, and `request.end` events.

## Hugging Face Space setup

1. Create a new Space.
2. Select `Docker` as SDK.
3. Push this repository to the Space.
4. In Space `Settings -> Variables and secrets`, add:
   - `GROQ_API_KEY` (if using Groq models)
   - `OPENAI_API_KEY` (if using OpenAI models)
5. Wait for build to complete.
6. Open the Space URL and test the UI.

---
title: Finance Decision Assistant
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
