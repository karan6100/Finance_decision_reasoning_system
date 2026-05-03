"""
Evaluation Dashboard — beginner-friendly visual eval for the Finance Assistant.

What this page does:
  1. Loads the 20 pre-built test cases from evals/eval_cases.jsonl
  2. Runs them through the live pipeline and measures latency
  3. Scores each case: routing, safety, reasoning, faithfulness (heuristic)
  4. Optional LLM-as-a-Judge mode for deeper scoring
  5. Shows a pass/fail summary table with color coding
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

# ── Path setup so this page can import src modules ──────────────────────────
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evals.eval_runner import (  # noqa: E402
    aggregate_results,
    build_snapshot,
    heuristic_scores,
    llm_judge_scores,
    load_jsonl,
)
from pipeline import run_finance_pipeline  # noqa: E402

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Eval Dashboard", page_icon=":test_tube:", layout="wide")
st.title("Evaluation Dashboard")
st.caption("Run the built-in test suite against the live pipeline and inspect quality scores.")

# ── Sidebar: API key (reuse what the user set on the Chat page) ──────────────
with st.sidebar:
    st.header("Configuration")

    provider = st.selectbox("LLM Provider", ["groq", "openai"], index=0, key="eval_provider")

    if provider == "groq":
        key_label, env_var = "Groq API Key", "GROQ_API_KEY"
        st.markdown("Free key at [console.groq.com](https://console.groq.com)")
    else:
        key_label, env_var = "OpenAI API Key", "OPENAI_API_KEY"

    api_key_input = st.text_input(key_label, type="password", placeholder="Paste your key here", key="eval_api_key")
    if api_key_input:
        os.environ[env_var] = api_key_input
        os.environ["LLM_PROVIDER"] = provider
        st.success("API key set.")

    st.divider()
    st.subheader("Eval Options")
    use_llm_judge = st.toggle(
        "LLM-as-a-Judge",
        value=False,
        help="Uses a second LLM call to score each response. More accurate but slower and uses API credits.",
    )
    max_cases = st.slider("Max cases to run", min_value=1, max_value=20, value=5,
                          help="Start small (3-5) to keep it fast.")

api_ready = bool(api_key_input) or bool(os.getenv("GROQ_API_KEY")) or bool(os.getenv("OPENAI_API_KEY"))

# ── Load eval cases ──────────────────────────────────────────────────────────
CASES_PATH = SRC_DIR / "evals" / "eval_cases.jsonl"
try:
    all_cases = load_jsonl(CASES_PATH)
except FileNotFoundError:
    st.error(f"Eval cases file not found at `{CASES_PATH}`. Make sure `evals/eval_cases.jsonl` exists.")
    st.stop()

cases_to_run = all_cases[:max_cases]

# ── Preview table ─────────────────────────────────────────────────────────────
with st.expander(f"Preview test cases ({len(cases_to_run)} of {len(all_cases)} selected)", expanded=False):
    st.dataframe(
        [
            {
                "ID": c["id"],
                "Query": c["query"],
                "Expected Route": c.get("expected_route", ""),
                "Expected Risk": c.get("expected_risk", ""),
                "Expected Quality": c.get("expected_quality", ""),
                "Tags": ", ".join(c.get("tags", [])),
            }
            for c in cases_to_run
        ],
        use_container_width=True,
    )

# ── What each metric means ────────────────────────────────────────────────────
with st.expander("What do the scores mean?", expanded=False):
    st.markdown("""
| Metric | What it checks |
|--------|---------------|
| **Routing** | Did the pipeline choose the right route? (basic / reasoning / clarification / unsafe) |
| **Risk** | Did the pipeline correctly classify risk? (low / medium / high) |
| **Safety** | Does the response avoid dangerous advice like "guaranteed returns"? |
| **Reasoning** | Does the response show structured analytical thinking? |
| **Factuality** | Is the response plausible and non-fabricated? |
| **Faithfulness** | Does the final response stay aligned with the internal candidate response? |
| **Clarity** | Is the response well-structured and readable? |
| **Governance** | Average of Routing + Risk accuracy |
| **Overall** | Weighted composite of all metrics |
| **Latency** | Time taken end-to-end by the pipeline (ms) |

**Scoring method:**
- **Heuristic** (default) — fast rule-based checks, no extra API calls
- **LLM Judge** — a second LLM call reads the full response and scores it using the rubric above
""")

# ── Run button ────────────────────────────────────────────────────────────────
if not api_ready:
    st.info("Enter your API key in the sidebar to run the evaluation.")
    st.stop()

if st.button("Run Evaluation", type="primary", use_container_width=True):

    llm_judge = None
    if use_llm_judge:
        try:
            from LLMs.llm_factory import LLMFactory  # noqa: E402
            llm_judge = LLMFactory.get_llm(provider=provider, task_type="complex")
        except Exception as exc:
            st.warning(f"LLM judge unavailable ({exc}). Falling back to heuristic scoring.")

    results: list[dict] = []
    progress = st.progress(0, text="Starting evaluation...")
    status_box = st.empty()

    for i, case in enumerate(cases_to_run):
        query = case.get("query", "").strip()
        status_box.markdown(f"**Running case {i + 1}/{len(cases_to_run)}:** `{query[:80]}`")

        t0 = time.perf_counter()
        error = ""
        try:
            state = run_finance_pipeline(query)
            snapshot = build_snapshot(state)
        except Exception as exc:
            error = str(exc)
            from evals.eval_runner import EvalSnapshot  # noqa: E402
            snapshot = EvalSnapshot(
                route=None, risk=None, final_response="", candidate_response="",
                valid_status=None, validation_reason="", attempt_counter=0, max_attempts=0,
            )

        runtime_ms = round((time.perf_counter() - t0) * 1000, 2)
        heuristic = heuristic_scores(case, snapshot)
        final_scores = heuristic

        if llm_judge and not error:
            try:
                final_scores = llm_judge_scores(llm_judge, case, snapshot, heuristic)
            except Exception as exc:
                heuristic["notes"] = list(heuristic.get("notes", [])) + [f"LLM judge failed: {exc}"]
                final_scores = heuristic

        results.append({
            "id": case.get("id"),
            "query": query,
            "expected_route": case.get("expected_route"),
            "actual_route": snapshot.route,
            "route_match": snapshot.route == case.get("expected_route"),
            "expected_risk": case.get("expected_risk"),
            "actual_risk": snapshot.risk,
            "risk_match": snapshot.risk == case.get("expected_risk"),
            "expected_quality": case.get("expected_quality"),
            "scores": final_scores,
            "runtime_ms": runtime_ms,
            "final_response": snapshot.final_response[:300],
            "error": error,
        })
        progress.progress((i + 1) / len(cases_to_run), text=f"{i + 1}/{len(cases_to_run)} done")

    status_box.empty()
    progress.empty()

    # ── Summary metrics ───────────────────────────────────────────────────────
    summary = aggregate_results(results)
    latencies = [r["runtime_ms"] for r in results if not r["error"]]

    st.subheader("Overall Results")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Pass Rate", f"{summary['pass_rate'] * 100:.0f}%")
    col2.metric("Route Accuracy", f"{summary['route_accuracy'] * 100:.0f}%")
    col3.metric("Risk Accuracy", f"{summary['risk_accuracy'] * 100:.0f}%")
    col4.metric("Avg Score", f"{summary['averages'].get('overall', 0):.2f}")
    if latencies:
        col5.metric("Avg Latency", f"{sum(latencies)/len(latencies):.0f} ms")

    # ── Scores bar chart ──────────────────────────────────────────────────────
    st.subheader("Average Score by Metric")
    avg = summary.get("averages", {})
    metric_order = ["routing", "risk", "safety", "reasoning", "factuality", "faithfulness", "clarity"]
    chart_data = {m.capitalize(): round(avg.get(m, 0), 3) for m in metric_order}
    st.bar_chart(chart_data)

    # ── Per-case results table ─────────────────────────────────────────────────
    st.subheader("Per-Case Results")
    table_rows = []
    for r in results:
        scores = r["scores"]
        table_rows.append({
            "ID": r["id"],
            "Query": r["query"][:60] + ("…" if len(r["query"]) > 60 else ""),
            "Expected Route": r["expected_route"],
            "Actual Route": r["actual_route"] or "ERROR",
            "Route ✓": "✅" if r["route_match"] else "❌",
            "Risk ✓": "✅" if r["risk_match"] else "❌",
            "Safety": round(scores.get("safety", 0), 2),
            "Reasoning": round(scores.get("reasoning", 0), 2),
            "Faithfulness": round(scores.get("faithfulness", 0), 2),
            "Overall": round(scores.get("overall", 0), 2),
            "Pass": "✅ PASS" if scores.get("pass") else "❌ FAIL",
            "Latency (ms)": r["runtime_ms"],
            "Error": r["error"][:60] if r["error"] else "",
        })

    st.dataframe(table_rows, use_container_width=True)

    # ── Failures detail ────────────────────────────────────────────────────────
    failures = [r for r in results if not r["scores"].get("pass")]
    if failures:
        with st.expander(f"Failed cases ({len(failures)})", expanded=True):
            for r in failures:
                st.markdown(f"**{r['id']}** — `{r['query']}`")
                st.markdown(f"- Route: expected `{r['expected_route']}` → got `{r['actual_route']}`")
                st.markdown(f"- Notes: {r['scores'].get('notes', [])}")
                if r["final_response"]:
                    st.markdown(f"- Response preview: _{r['final_response'][:200]}_")
                st.divider()

    # ── Latency breakdown ─────────────────────────────────────────────────────
    if latencies:
        with st.expander("Latency breakdown"):
            import statistics
            st.markdown(f"""
| Stat | Value |
|------|-------|
| Min | {min(latencies):.0f} ms |
| Max | {max(latencies):.0f} ms |
| Avg | {statistics.mean(latencies):.0f} ms |
| Median | {statistics.median(latencies):.0f} ms |
""")
            latency_chart = {r["id"]: r["runtime_ms"] for r in results if not r["error"]}
            st.bar_chart(latency_chart)

    # ── Export ─────────────────────────────────────────────────────────────────
    st.subheader("Export Results")
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Download per-case results (JSONL)",
            data="\n".join(json.dumps(r, default=str) for r in results),
            file_name="eval_results.jsonl",
            mime="application/jsonl",
        )
    with col_b:
        st.download_button(
            "Download summary (JSON)",
            data=json.dumps(summary, indent=2, default=str),
            file_name="eval_summary.json",
            mime="application/json",
        )
