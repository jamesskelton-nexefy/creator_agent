# LangGraph Storage Approach for Supabase

## Problem Summary

LangGraph has **two storage layers**:
1. **Thread/Assistant Registry** - Where threads are created and tracked
2. **Checkpointer** - Where conversation state is persisted

| Command | Thread Registry | Checkpointer | Custom PostgreSQL |
|---------|----------------|--------------|-------------------|
| `langgraph dev` | File-based (`.langgraph_api/`) | Your configured checkpointer | Thread registry: No, Checkpointer: Yes |
| `langgraph up` | PostgreSQL | PostgreSQL (auto-configured) | **Yes - uses `--postgres-uri`** |

## Solution Options

### Option 1: Use `langgraph up` (Recommended for Persistence)

This runs a Docker-based server that uses PostgreSQL for ALL storage.

**Requirements:**
- Docker Desktop installed and running
- Supabase local running on port 15322

**Start command:**
```bash
npx @langchain/langgraph-cli up --postgres-uri "postgresql://postgres:postgres@host.docker.internal:15322/postgres" --port 8000 --watch
```

Or use the provided batch file:
```bash
start-langgraph.bat
```

**Pros:**
- True persistence - threads survive server restarts
- All data in Supabase PostgreSQL
- Production-like environment

**Cons:**
- Requires Docker
- Slower startup (builds Docker image)
- More resource intensive

---

### Option 2: Continue with `langgraph dev` (Development Only)

Accept that `langgraph dev` is ephemeral by design. The thread registry resets on restart.

**Approach:**
1. Keep using `langgraph dev` for fast iteration
2. Clear `blueprints.chat_threads` on server restart
3. Your PostgresSaver checkpointer DOES persist checkpoint data
4. But thread lookup requires the thread to exist in LangGraph's registry

**Frontend Changes Needed:**
- Handle 404 errors gracefully when thread doesn't exist in LangGraph
- Auto-create thread in LangGraph when selecting from history
- Or: Clear chat history on page load if threads return 404

---

### Option 3: Hybrid Approach (Advanced)

Modify your frontend to be the source of truth:

1. **Frontend stores thread IDs** in `blueprints.chat_threads`
2. **On thread select**: Check if thread exists in LangGraph (GET /threads/{id}/state)
3. **If 404**: Create thread in LangGraph (POST /threads) with the same ID
4. **Checkpoints persist** via PostgresSaver, so state CAN be recovered

This requires the LangGraph CLI to support creating threads with specific IDs (check API).

---

## Current Architecture

```
┌─────────────────┐     ┌──────────────────────────────────────┐
│    Frontend     │     │         LangGraph CLI               │
│  (CopilotKit)   │     │                                      │
│                 │     │  ┌─────────────────────────────────┐ │
│ blueprints.     │────▶│  │  Thread Registry (dev mode)    │ │
│ chat_threads    │     │  │  .langgraph_api/ (ephemeral)   │ │
│ (Supabase)      │     │  └─────────────────────────────────┘ │
│                 │     │                 │                    │
│                 │     │  ┌─────────────────────────────────┐ │
│                 │     │  │  PostgresSaver (checkpointer)  │ │
│                 │     │  │  public.checkpoints table      │ │
│                 │────▶│  │  (persists, but needs thread)  │ │
│                 │     │  └─────────────────────────────────┘ │
└─────────────────┘     └──────────────────────────────────────┘
```

## Recommendation

**For Development:** Use `langgraph dev` and clear `blueprints.chat_threads` when server restarts.

**For Production/Staging:** Use `langgraph up --postgres-uri` pointing to your Supabase PostgreSQL.

## Notes

- The PostgresSaver checkpointer you configured in `node-expert-agent.ts` IS being used for checkpoint data
- The issue is that LangGraph CLI's thread registry is SEPARATE from the checkpointer
- `langgraph up` solves this by using PostgreSQL for EVERYTHING
- Your `blueprints.chat_threads` table was an attempt to persist thread metadata, but it's out of sync with LangGraph's actual thread registry









