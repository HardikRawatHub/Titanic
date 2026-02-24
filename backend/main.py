"""
main.py  –  FastAPI Backend
---------------------------
Exposes a single POST /chat endpoint consumed by the Streamlit frontend.
Also provides /health, /dataset/info, and /dataset/chart endpoints for
direct API access.

Run with:
    uvicorn main:app --reload --port 8000
"""

import json
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_loader import get_df, get_summary_stats
from visualizer import CHART_REGISTRY
from agent import run_agent

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Titanic Chat Agent API",
    description="LangChain-powered agent for Titanic dataset analysis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ───────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:      str
    chat_history: list[dict] = []   # [{"role": "human"|"ai", "content": str}]


class ChatResponse(BaseModel):
    text:        str
    image_b64:   Optional[str] = None
    chart_type:  Optional[str] = None
    latency_ms:  int


class ChartRequest(BaseModel):
    chart_type: str


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Quick health-check endpoint."""
    return {"status": "ok", "message": "Titanic Agent API is running 🚢"}


@app.get("/dataset/info")
def dataset_info():
    """Return dataset metadata and summary statistics."""
    df    = get_df()
    stats = get_summary_stats()
    missing = df.isnull().sum()
    missing = {k: int(v) for k, v in missing.items() if v > 0}
    return {
        "rows":           len(df),
        "columns":        list(df.columns),
        "missing_values": missing,
        "summary":        stats,
        "available_charts": list(CHART_REGISTRY.keys()),
    }


@app.post("/dataset/chart")
def get_chart(req: ChartRequest):
    """
    Generate a specific chart and return it as a base64 PNG.
    Useful for direct chart access without going through the agent.
    """
    if req.chart_type not in CHART_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown chart '{req.chart_type}'. "
                   f"Available: {list(CHART_REGISTRY.keys())}"
        )
    df  = get_df()
    fn  = CHART_REGISTRY[req.chart_type]
    b64 = fn(df)
    return {"chart_type": req.chart_type, "image_b64": b64}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main chat endpoint consumed by Streamlit.
    Passes the user message through the LangChain agent and returns
    the text answer plus an optional base64 chart image.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Convert chat history format for LangChain
    lc_history = []
    for msg in req.chat_history:
        if msg.get("role") == "human":
            lc_history.append(("human", msg["content"]))
        elif msg.get("role") == "ai":
            lc_history.append(("ai", msg["content"]))

    t0 = time.time()
    try:
        result = run_agent(req.message, lc_history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    latency = int((time.time() - t0) * 1000)
    return ChatResponse(
        text=result["text"],
        image_b64=result.get("image_b64"),
        chart_type=result.get("chart_type"),
        latency_ms=latency,
    )


# ── Dev entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
