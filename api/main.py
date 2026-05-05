import json
import uuid
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import AsyncConnectionPool
import aiosqlite

from api.schema import ChatRequest, ChatResponse

# Global graph instance — initialized once at startup
agent_graph = None


# ── Lifespan — runs once at startup and shutdown ───────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes the graph once at startup using AsyncSqliteSaver.
    Keeps it alive across all requests.
    Shuts it down cleanly when server stops.
    """
    global agent_graph

    from src.config import Config
    from src.multi_agent_graph import MultiAgentAgricultureGraph

    config = Config()
    print(f"Starting up: Initializing Agri-Agent Graph (Mode: {config.PERSISTENCE_MODE})...")

    try:
        if config.PERSISTENCE_MODE == "postgres":
            # Industrial Postgres Checkpointer
            async with AsyncConnectionPool(conninfo=config.DATABASE_URL, max_size=20) as pool:
                async with PostgresSaver(pool) as saver:
                    # Initialize tables if they don't exist
                    await saver.setup()
                    manager     = MultiAgentAgricultureGraph(checkpointer=saver)
                    agent_graph = manager.app
                    print("Graph initialized with PostgresSaver.")
                    yield
        else:
            # Local SQLite Checkpointer
            async with AsyncSqliteSaver.from_conn_string(config.CHECKPOINT_DB_PATH) as saver:
                manager     = MultiAgentAgricultureGraph(checkpointer=saver)
                agent_graph = manager.app
                print("Graph initialized with AsyncSqliteSaver.")
                yield

    except Exception as e:
        # Fallback: use MemorySaver if Checkpointers fail
        print(f"Checkpointer failed ({e}), falling back to MemorySaver...")
        from src.multi_agent_graph import _build_graph
        agent_graph = _build_graph(use_checkpointer=True)
        yield

    print("Shutting down...")


# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(
    title="Agri-Agent API",
    description="REST API for the Agricultural Multi-Agent System",
    version="1.0.0",
    lifespan=lifespan
)


# ── POST /chat ─────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint — returns the final answer after full agent execution.

    Uses a fresh thread_id derived from the request thread_id to avoid
    KeyError: '__start__' in LangGraph 1.x MemorySaver/SqliteSaver.
    """
    if not agent_graph:
        raise HTTPException(status_code=500, detail="Agent graph not initialized")

    # Generate fresh thread_id per request to avoid __start__ KeyError
    # Append uuid suffix so each call gets a clean checkpoint
    fresh_thread_id = f"{request.thread_id}-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": fresh_thread_id}}
    inputs = {"messages": [HumanMessage(content=request.message)]}

    try:
        final_answer = ""

        # Use astream with stream_mode="values" to get full state at each step
        # This works correctly with AsyncSqliteSaver
        async for state in agent_graph.astream(
            inputs,
            config=config,
            stream_mode="values"
        ):
            messages = state.get("messages", [])
            if not messages:
                continue

            last_msg = messages[-1]

            # Capture the last non-tool-call AIMessage as the final answer
            if (isinstance(last_msg, AIMessage)
                    and last_msg.content
                    and not last_msg.tool_calls):
                final_answer = last_msg.content

        if not final_answer:
            final_answer = "Agent completed but produced no text response."

        return ChatResponse(
            answer=final_answer,
            thread_id=request.thread_id,   # return original thread_id to client
            status="completed"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /stream ───────────────────────────────────────────────
@app.post("/stream")
async def stream_endpoint(request: ChatRequest):
    """
    Streaming endpoint using Server-Sent Events (SSE).
    Yields node-by-node updates as the agent executes.
    Frontend receives chunks in real time — ChatGPT-like experience.
    """
    if not agent_graph:
        raise HTTPException(status_code=500, detail="Agent graph not initialized")

    async def event_generator():
        # Fresh thread_id per stream request
        fresh_thread_id = f"{request.thread_id}-{uuid.uuid4()}"
        config = {"configurable": {"thread_id": fresh_thread_id}}
        inputs = {"messages": [HumanMessage(content=request.message)]}

        try:
            async for event in agent_graph.astream(
                inputs,
                config=config,
                stream_mode="updates"
            ):
                for node_name, output in event.items():
                    # Extract readable content from output
                    messages = output.get("messages", [])
                    content  = ""
                    if messages:
                        last = messages[-1]
                        if hasattr(last, "content"):
                            content = last.content[:300]

                    data = {
                        "node":    node_name,
                        "status":  "processing",
                        "content": content,
                        "data":    str(output)[:200]
                    }
                    yield {
                        "event": "update",
                        "id":    str(asyncio.get_event_loop().time()),
                        "data":  json.dumps(data)
                    }

            # Signal stream completion
            yield {
                "event": "done",
                "data":  json.dumps({"status": "completed"})
            }

        except Exception as e:
            yield {
                "event": "error",
                "data":  json.dumps({"detail": str(e)})
            }

    return EventSourceResponse(event_generator())


# ── GET / ──────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Root endpoint — confirms API is running."""
    return {
        "status":  "running",
        "service": "Agri-Agent API",
        "version": "1.0.0",
        "endpoints": {
            "chat":   "POST /chat",
            "stream": "POST /stream",
            "docs":   "GET /docs"
        }
    }


# ── Direct run ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)