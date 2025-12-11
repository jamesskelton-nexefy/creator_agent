@echo off
echo Clearing LangGraph file cache...
if exist ".langgraph_api" (
    rmdir /s /q ".langgraph_api"
    echo Cache cleared.
) else (
    echo No cache to clear.
)
echo Starting LangGraph server...
npx @langchain/langgraph-cli dev --host localhost --port 8000
