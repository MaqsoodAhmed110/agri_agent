# Implementation Plan - Fix KeyError: '__start__' and Restore Persistence

The user is experiencing a `KeyError: '__start__'` in LangGraph, which is a known issue when using `MemorySaver` in environments with complex lifecycles like Streamlit. Additionally, the current "fix" in the codebase (generating a fresh `thread_id` per query) breaks conversation persistence.

## User Review Required

> [!IMPORTANT]
> This change will switch the checkpointing mechanism from **In-Memory** (`MemorySaver`) to **SQLite** (`SqliteSaver`). This means a file named `checkpoint_db.sqlite` will be created in the project root to store conversation states. This is necessary for stable Human-in-the-Loop (HITL) and persistence.

## Proposed Changes

### 1. Upgrade Persistence to `SqliteSaver`
Switching from `MemorySaver` to `SqliteSaver` provides a more robust, file-based persistence that survives Streamlit reruns and avoids the in-memory state corruption that causes `KeyError: '__start__'`.

#### [MODIFY] [multi_agent_graph.py](file:///f:/favourtes/Maqsood%20%20Data/GIKI%207Sem/Agentic%20Elective/agriculture_agent/src/multi_agent_graph.py)
- Import `SqliteSaver` from `langgraph.checkpoint.sqlite`.
- Update `MultiAgentAgricultureGraph` to use `SqliteSaver` with the path from `Config`.
- Use `START` explicitly in the graph definition.

#### [MODIFY] [main.py](file:///f:/favourtes/Maqsood%20%20Data/GIKI%207Sem/Agentic%20Elective/agriculture_agent/app/main.py)
- Remove the "fresh thread ID per query" logic.
- Use a single `thread_id` per conversation session.
- Ensure the stable `thread_id` is used for both `stream` and `invoke` (resume).

### 2. State & Node Handling
- Ensure `AgentState` fields are handled consistently.
- Verify `ToolNode` integration with the checkpointer.

## Verification Plan

### Automated Tests
- Run a updated `scratch/repro_error.py` to ensure the graph runs with persistence.
- Run `src/persistence_test.py` to verify cross-turn memory.

### Manual Verification
1. Run `streamlit run app/main.py`.
2. Send an initial query: "What are the common challenges for wheat in Punjab?".
3. Send a follow-up: "How can I fix the first one?".
4. Verify the agent remembers context.
5. Trigger an email action and verify "Proceed" works.
