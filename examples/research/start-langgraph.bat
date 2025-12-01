@echo off
REM Start LangGraph server with Supabase PostgreSQL storage
REM This uses Docker-based deployment for proper persistence

REM Supabase local PostgreSQL connection
SET POSTGRES_URI=postgresql://postgres:postgres@host.docker.internal:15322/postgres

echo Starting LangGraph with Supabase PostgreSQL storage...
echo PostgreSQL: %POSTGRES_URI%
echo.

REM Use langgraph up for Docker-based deployment with PostgreSQL
npx @langchain/langgraph-cli up ^
  --postgres-uri "%POSTGRES_URI%" ^
  --port 8000 ^
  --watch

pause

