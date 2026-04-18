#!/usr/bin/env bash
set -euo pipefail

uvicorn backend_api:app --host 0.0.0.0 --port 8000 &

exec streamlit run streamlit_app.py \
  --server.port=7860 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false
