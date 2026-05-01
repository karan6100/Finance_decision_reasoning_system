#!/usr/bin/env bash
set -e

exec streamlit run streamlit_app.py \
  --server.port=7860 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false
