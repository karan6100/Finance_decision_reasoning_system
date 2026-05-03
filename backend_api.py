import logging
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from logging_config import configure_logging, get_request_id, reset_request_id, set_request_id
from pipeline import run_finance_pipeline


class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)


class AnalyzeResponse(BaseModel):
    response: str
    route: Optional[str] = None
    risk: Optional[str] = None
    reasons: list[str] = Field(default_factory=list)


app = FastAPI(
    title="Finance Decision Reasoning API",
    description="Backend API for finance query analysis",
    version="1.0.0",
)

configure_logging()
logger = logging.getLogger(__name__)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    token = set_request_id(request_id)
    started_at = time.perf_counter()

    logger.info(
        "request.start",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else None,
        },
    )

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.exception(
            "request.error",
            extra={
                "method": request.method,
                "path": request.url.path,
                "duration_ms": duration_ms,
            },
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request.end",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response
    finally:
        reset_request_id(token)


@app.get("/health")
def health() -> dict[str, str]:
    logger.debug("health.check")
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    try:
        logger.info(
            "analyze.started",
            extra={
                "query_length": len(payload.query),
            },
        )
        state = run_finance_pipeline(payload.query)
        final_response = state.get("final_response", "")
        if not final_response:
            raise RuntimeError("Pipeline completed without a final response.")

        profile = state.get("profile")
        risk = getattr(profile, "risk", None)
        logger.info(
            "analyze.completed",
            extra={
                "route": state.get("route"),
                "risk": risk,
                "reasons_count": len(state.get("reasons", [])),
            },
        )

        return AnalyzeResponse(
            response=final_response,
            route=state.get("route"),
            risk=risk,
            reasons=state.get("reasons", []),
        )
    except Exception:
        logger.exception("analyze.failed")
        request_id = get_request_id()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze query. request_id={request_id}",
        ) from None
