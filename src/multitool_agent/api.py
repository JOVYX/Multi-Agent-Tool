"""FastAPI web service exposing the Multi-Tool AI Agent via REST and SSE streaming."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from multitool_agent.agent import Agent
from multitool_agent.config import get_settings
from multitool_agent.models import AgentResponse, ToolCall, ToolResult
from multitool_agent.utils.logging import get_logger

logger = get_logger("multitool_agent.api")

app = FastAPI(
    title="Multi-Tool AI Agent API",
    description="Production REST and Server-Sent Events (SSE) API for an autonomous tool-calling AI agent.",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (session_id -> Agent instance)
_SESSIONS: Dict[str, Agent] = {}


def get_or_create_agent(session_id: Optional[str] = None) -> tuple[str, Agent]:
    """Retrieve an existing agent session or instantiate a new one."""
    sid = session_id or str(uuid4())
    if sid not in _SESSIONS:
        logger.info(f"Creating new agent session: {sid}")
        _SESSIONS[sid] = Agent()
    return sid, _SESSIONS[sid]


# --- Request & Response Schemas ---


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User instruction or question for the agent.")
    session_id: Optional[str] = Field(None, description="Optional persistent session identifier for multi-turn chat.")


class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class ToolCallData(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any]


class ToolResultData(BaseModel):
    tool_use_id: str
    content: str
    is_error: bool


class StepData(BaseModel):
    iteration: int
    tool_calls: List[ToolCallData]
    tool_results: List[ToolResultData]
    assistant_text: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    iterations: int
    steps: List[StepData]
    finish_reason: str


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    tools_count: int


# --- API Endpoints ---


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Return service health and current model provider configuration."""
    settings = get_settings()
    active_model = settings.groq_model if settings.llm_provider == "groq" else settings.anthropic_model
    dummy_agent = Agent()
    return HealthResponse(
        status="healthy",
        provider=settings.llm_provider,
        model=active_model,
        tools_count=len(dummy_agent.registry.list_tools()),
    )


@app.get("/api/v1/tools", response_model=List[ToolInfo], tags=["Tools"])
def list_tools() -> List[ToolInfo]:
    """Return all tools registered in the agent's registry along with their JSON schemas."""
    agent = Agent()
    tools: List[ToolInfo] = []
    for tool in agent.registry.list_tools():
        schema = tool.schema()
        tools.append(
            ToolInfo(
                name=tool.name,
                description=tool.description,
                parameters=schema.get("input_schema", {}),
            )
        )
    return tools


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest) -> ChatResponse:
    """Execute a complete agent reasoning run synchronously and return the final response."""
    session_id, agent = get_or_create_agent(request.session_id)
    try:
        agent_resp: AgentResponse = agent.run(request.message)
    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent reasoning failed: {str(e)}",
        )

    steps_data: List[StepData] = []
    for step in agent_resp.steps:
        steps_data.append(
            StepData(
                iteration=step.iteration,
                tool_calls=[ToolCallData(id=tc.id, name=tc.name, arguments=tc.arguments) for tc in step.tool_calls],
                tool_results=[
                    ToolResultData(tool_use_id=tr.tool_use_id, content=tr.content, is_error=tr.is_error)
                    for tr in step.tool_results
                ],
                assistant_text=step.assistant_text,
            )
        )

    return ChatResponse(
        session_id=session_id,
        response=agent_resp.text,
        iterations=agent_resp.iterations,
        steps=steps_data,
        finish_reason=agent_resp.finish_reason,
    )


@app.post("/api/v1/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest) -> EventSourceResponse:
    """Stream agent execution events and generated response tokens using Server-Sent Events (SSE)."""
    session_id, agent = get_or_create_agent(request.session_id)
    queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    def on_token(chunk: str) -> None:
        queue.put_nowait({"event": "token", "data": {"text": chunk}})

    def run_agent_in_thread() -> None:
        try:
            resp = agent.run(request.message, stream=True, stream_callback=on_token)
            queue.put_nowait(
                {
                    "event": "done",
                    "data": {
                        "session_id": session_id,
                        "text": resp.text,
                        "iterations": resp.iterations,
                    },
                }
            )
        except Exception as exc:
            logger.error(f"Streaming error: {exc}", exc_info=True)
            queue.put_nowait({"event": "error", "data": {"error": str(exc)}})
        finally:
            queue.put_nowait({"event": "end", "data": {}})

    async def event_generator() -> AsyncGenerator[Dict[str, Any], None]:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, run_agent_in_thread)

        while True:
            item = await queue.get()
            if item["event"] == "end":
                break
            yield {
                "event": item["event"],
                "data": json.dumps(item["data"]),
            }

    return EventSourceResponse(event_generator())


@app.delete("/api/v1/sessions/{session_id}", tags=["Chat"])
def clear_session(session_id: str) -> Dict[str, str]:
    """Reset and remove an active session's conversation history."""
    if session_id in _SESSIONS:
        _SESSIONS[session_id].clear_history()
        del _SESSIONS[session_id]
        return {"status": "cleared", "session_id": session_id}
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
