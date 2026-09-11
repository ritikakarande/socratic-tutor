"""FastAPI application entry point.

Run with::

    uvicorn app.main:app --reload

Logging is configured from settings at startup, and the tutor agent is warmed
up so the first request is fast. Missing API keys never prevent startup; the
system falls back to offline components and reports its state via /health.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import routes
from app.config.settings import get_settings
from app.utils.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(level=settings.log_level, as_json=settings.log_json)
logger = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Socratic Tutor API")
    logger.info("Configuration: %s", settings.redacted())
    routes.get_agent()  # warm up
    yield
    logger.info("Shutting down Socratic Tutor API")


app = FastAPI(
    title="Socratic Tutor API",
    description="An agentic, Socratic math and science tutor with RAG and guardrails.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(routes.router)


@app.get("/")
def root() -> dict:
    return {
        "name": "Socratic Tutor API",
        "docs": "/docs",
        "health": "/health",
    }
