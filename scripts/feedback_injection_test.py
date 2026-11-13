"""
Feedback-injection (canary) test for the validator-feedback retry loop.

QUESTION THIS ANSWERS
    "When the validator rejects an answer, does the REAL reasoning model actually
    follow the validator's feedback in its next answer?"

WHAT IT DOES
    Replaces the validator with a fake that rejects answer 1 with a "canary"
    instruction ("begin with a one-line summary that starts with 'Bottom line:'")
    and passes answer 2. A canary is a marker the model would almost never write
    unprompted, so it can only show up in answer 2 if the feedback got through.
    For each reasoning-route query from evals/eval_cases.jsonl it checks:
        answer 1 has no canary  AND  answer 2 has the canary  ->  FOLLOWED
    The reasoning LLM is REAL (about 2 calls per query, needs an API key); only the
    validator is faked. Results vary run to run, so this is a script, not a pytest test.

WHAT IT DOES *NOT* ANSWER
    - Whether the model fixes REAL issues well (guarantees, missing caveats, fabricated
      figures). The canary proves the feedback channel works, not answer quality.
      -> realistic injection + LLM judge / eval retry metrics
    - Whether the feedback is wired into the prompt correctly in every edge case
      (first attempt, pass, parse errors).
      -> tests/test_retry_prompt_capture.py (free, deterministic, runs in pytest)
    - How the REAL validator behaves; it is replaced by the fake.

WHY IT EXISTS
    The prompt-capture tests prove the reason is IN the prompt; this proves the model
    ACTS on it. Comparing the current code with the pre-change code (commit 75a48fe,
    where the retry prompt never contained feedback) gives a before/after number,
    e.g. "followed injected feedback in N/N retries vs 0/N before". Only quote numbers
    you have actually measured.

HOW TO RUN (from the repo root)
    python test_scripts/feedback_injection_test.py                         # current code
    python test_scripts/feedback_injection_test.py --show                  # also print answer previews
    python test_scripts/feedback_injection_test.py --max-cases 11          # all reasoning-route cases

    # Control run against the pre-change code (commit 75a48fe):
    git worktree add ../baseline 75a48fe
    cp .env ../baseline/                                                   # .env isn't tracked by git
    python test_scripts/feedback_injection_test.py --src ../baseline/src
    git worktree remove --force ../baseline                                # --force: copied .env is untracked
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = REPO_ROOT / "src"
CANARY = "Bottom line:"
FAKE_REASON = f"The answer must begin with a one-line summary that starts with '{CANARY}'."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that validator feedback reaches the retry prompt.")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC,
                        help="Path to the src/ directory to test (default: this repo's src/).")
    parser.add_argument("--cases", type=Path, default=None,
                        help="JSONL eval cases (default: <src>/evals/eval_cases.jsonl).")
    parser.add_argument("--max-cases", type=int, default=5,
                        help="Max allow_reasoning cases to run (default: 5).")
    parser.add_argument("--show", action="store_true",
                        help="Print the first 300 chars of each answer.")
    return parser.parse_args()


def load_reasoning_queries(cases_path: Path, limit: int) -> list[tuple[str, str]]:
    queries = []
    with cases_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("expected_route") == "allow_reasoning":
                queries.append((row.get("id", "?"), row["query"]))
    return queries[:limit]


def main() -> None:
    args = parse_args()
    src_dir = args.src.resolve()
    if not (src_dir / "pipeline.py").exists():
        sys.exit(f"pipeline.py not found under {src_dir}; pass --src <path-to-src>")

    # Make the target src/ importable before importing any project modules.
    sys.path.insert(0, str(src_dir))
    from logging_config import configure_logging  # noqa: E402
    configure_logging()                         # honours LOG_FORMAT / LOG_FILE / LOG_LEVEL
    import nodes.validation_node as vn          # noqa: E402
    from pipeline import run_finance_pipeline   # noqa: E402

    calls: list[str] = []

    def fake_validator(inp):
        calls.append(inp.candidate_response)
        if len(calls) == 1:
            return {"decision": "fail", "reason": FAKE_REASON}
        return {"decision": "pass", "reason": "ok"}

    # validation_node looks up this module-level instance on every call.
    vn.validation_agent.run = fake_validator

    cases_path = args.cases or (src_dir / "evals" / "eval_cases.jsonl")
    queries = load_reasoning_queries(cases_path, args.max_cases)
    if not queries:
        sys.exit(f"No allow_reasoning cases found in {cases_path}")

    print(f"Testing code in: {src_dir}")
    print(f"Canary: {CANARY!r}\n")

    tested = followed = chance = 0
    for case_id, query in queries:
        calls.clear()
        try:
            state = run_finance_pipeline(query)
        except Exception as exc:  # LLM/rate-limit errors shouldn't stop the run
            print(f"[{case_id}] ERROR   | {type(exc).__name__}: {exc}")
            continue

        route = state.get("route")
        if route != "allow_reasoning" or len(calls) < 2:
            print(f"[{case_id}] SKIPPED | route={route}, validator calls={len(calls)} | {query[:50]}")
            continue

        a1, a2 = calls[0], calls[1]
        a1_hit = CANARY.lower() in a1.lower()
        a2_hit = CANARY.lower() in a2.lower()
        tested += 1
        followed += a2_hit and not a1_hit
        chance += a1_hit

        verdict = "FOLLOWED" if (a2_hit and not a1_hit) else "IGNORED " if not a2_hit else "UNCLEAR "
        print(f"[{case_id}] {verdict}| ans1 canary={a1_hit!s:5} ans2 canary={a2_hit!s:5} | {query[:50]}")
        if args.show:
            print(f"    ans1: {a1[:300]!r}")
            print(f"    ans2: {a2[:300]!r}\n")

    print(f"\nFeedback followed: {followed}/{tested}")
    if chance:
        print(f"Warning: canary appeared in {chance} first answer(s) without feedback; pick a rarer canary.")


if __name__ == "__main__":
    main()
