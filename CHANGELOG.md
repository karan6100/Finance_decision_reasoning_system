# Changelog

All notable changes to the Finance Decision Assistant are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Entries are grouped by date, newest first.

## [Unreleased]

### Known issues
- Retries on the basic (low-risk) path resend the same prompt; they don't yet receive validator feedback.
- A provider error during an LLM call (rate limit, timeout) stops the pipeline instead of returning a clear message to the user.

## 2026-11-12

### Added
- **Validator feedback on retries.** When an answer fails validation, the next reasoning attempt receives the rejected answer and the validator's reason, so it can fix the specific issues instead of starting over blind.
- **Rate limiting for LLM calls.** All model calls share one rate limiter (default: one call every 2 seconds, no bursts) to avoid hitting free-tier provider limits. Configurable with the `LLM_MIN_INTERVAL_SECONDS` environment variable (the value is in requests per second).
- **Dedicated judge model.** LLM-as-a-Judge scoring now uses its own `judge` entry in `model_configs.yaml`, so the judge can be set separately from the answer generator. On Groq it uses a different model family, which reduces self-preference bias.
- **Length-bias guidance for the judge.** The judge rubric now tells the model not to reward long answers on their own and to penalize padding.

### Changed
- **Groq models updated:** `openai/gpt-oss-20b` for basic queries and `openai/gpt-oss-120b` for reasoning and complex queries.
- `--judge-provider` in the eval CLI no longer restricts the choice to `groq` / `openai`.

### Fixed
- **Every generated answer is now validated.** Previously the last attempt was returned without being checked (5 answers generated, only 4 validated).
- **Validator glitches are no longer treated as wrong answers.** If the validator returns empty or malformed output, the same answer is checked again (up to 2 times) instead of being thrown away. If the validator keeps failing, the run stops safely and returns no unverified answer.
- A crash on the first reasoning call caused by reading validation feedback that didn't exist yet.

## 2026-05-03

### Changed
- Consolidated the app entry point into `src/streamlit_app.py`.
- Moved `eval_runner.py` into `src/evals/` next to the eval cases.
