#!/usr/bin/env python
"""
title: B2CC MCP + vLLM Agent
author: Dhiraj
description: LangGraph + MCP agent with tool-call execution
licence: MIT
"""

import asyncio, re, json
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# -----------------------------
# Vector DB (Chroma retriever)
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb = Chroma(persist_directory="./data/chroma_db", embedding_function=embeddings)
retriever = vectordb.as_retriever()

# -----------------------------
# MCP Client (MySQL + GitLab)
# -----------------------------
client = MultiServerMCPClient(
    {
        "mysql": {
            "url": "http://localhost:9001/mcp/",
            "transport": "streamable_http",
        },
        "gitlab": {
            "url": "http://localhost:9002/mcp/",
            "transport": "streamable_http",
        },
    }
)

# -----------------------------
# Example MySQL schema prompt
# -----------------------------
MYSQL_SCHEMA = """
You are connected to a MySQL MCP server.

## Database Schema
The following tables and columns are available. You must only use these:

- am_disk_usage(id, created_at, mount_point, used, available, total_size)
- fm_alarm_record(id, alarm_code, severity, t)
- misc_data(id, key, value, t)
- raw_datasets(id, dataset, nbytes, created_at, expno, runno, nfiles, belle2_release)

## Rules for Query Generation
1. **Schema Adherence**
   - Only reference the tables and columns listed above.
   - Do not invent or assume additional tables/columns.

2. **SQL Style**
   - Always generate complete, valid SQL queries.
   - Prefer explicit column names over `SELECT *` unless all columns are required.
   - Add `ORDER BY`, `LIMIT`, or `WHERE` filters when appropriate to keep results concise.

3. **Execution**
   - Always run the SQL queries using the `mysql-mcp` tool.
   - After execution, format the results into **human-readable explanations** (tables, summaries, or descriptive text).

4. **Schema Requests**
   - If the user asks about "tables" or "schema," respond only with this context (do not query the DB).

5. **Interpretation**
   - When the user asks a question, first determine which table(s) and column(s) are relevant.
   - If multiple interpretations are possible, choose the one most consistent with the schema.

6. **Output**
   - Present query results in clear natural language with supporting tables when needed.
   - Avoid dumping raw SQL results without explanation.
"""


# -----------------------------
# Example GitLab schema prompt
# -----------------------------
GITLAB_SCHEMA = """
You are connected to a GitLab MCP server.

## Capabilities
You can query the following resources:
- **Projects**: metadata about GitLab projects.
- **Issues (Tickets)**: id, title, description, state (open/closed), labels, assignees, created_at, updated_at, web_url.
- **Merge Requests (MRs)**: id, title, description, state, author, created_at, updated_at, web_url.
- **Repository Files / Docs**: file names, paths, and web_url.

## Rules for Query Generation
1. **Schema Adherence**
   - Only query issues, merge requests, projects, or repo docs.
   - Use `web_url` when available to provide clickable links.

2. **Query Style**
   - Be explicit: filter by `state`, `labels`, `assignee`, or `date` when possible.
   - Use ordering (`created_at`, `updated_at`, `merged_at`) or limits to keep results concise.
   - Always include the **ID and web_url** in responses.

3. **Execution**
   - Always run the query using the `gitlab-mcp` tool.
   - Format results into **human-readable summaries** (tables, bullet points, or narrative).

4. **Interpretation**
   - Decide whether the user is asking about tickets, MRs, or repo docs.
   - If ambiguous, assume they mean **issues/tickets**.

5. **Output**
   - Summarize clearly in natural language.
   - Provide **titles, state, timestamps, and links** at minimum.
   - Avoid raw JSON dumps unless explicitly requested.
"""

# -----------------------------
# Helper: doc answers
# -----------------------------
async def answer_from_docs(llm, question, docs):
    context = "\n\n".join(d.page_content for d in docs[:3])
    prompt = f"""
You are a helpful assistant.
The user asked: {question}

Here are the top retrieved documents:

{context}

Please provide a clear, human-like answer using these docs as context.
If the docs do not fully answer, politely say so.
"""
    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    return resp.content

# -----------------------------
# Router
# -----------------------------
def route_query(user_input: str):
    text = user_input.lower()
    if any(word in text for word in ["table", "schema", "sql", "database", "runs", "disk_space", "alerts", "db", "query"]):
        return "mysql"
    elif any(word in text for word in ["ticket", "issue", "merge", "gitlab"]):
        return "gitlab"
    elif any(word in text for word in ["doc", "guide", "setup", "install", "gbasf2"]):
        return "retriever"
    return "general"

# -----------------------------
# Normalize messages
# -----------------------------
def normalize_messages(messages):
    safe = []
    for i, m in enumerate(messages):
        role = getattr(m, "role", None)
        if i == 0 and isinstance(m, SystemMessage):
            safe.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            safe.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            safe.append({"role": "assistant", "content": m.content})
        else:
            safe.append({"role": "assistant", "content": f"[{role}] {m.content}"})
    return safe

# -----------------------------
# Tool-call handler
# -----------------------------
async def handle_tool_calls(agent, client, memory, result):
    """
    Handle one or more tool calls until no <tool_call> blocks remain.
    """
    messages = result["messages"]
    last_ai = [m for m in messages if isinstance(m, AIMessage)]
    if not last_ai:
        return str(result)

    answer = last_ai[-1].content

    while "<tool_call>" in answer:
        match = re.search(r"<tool_call>\s*(\{.*\})\s*</tool_call>", answer, re.DOTALL)
        if not match:
            break

        try:
            tool_call = json.loads(match.group(1))
            tool_name = tool_call["name"]
            args = tool_call.get("arguments", {})

            print(f"⚙️ Executing tool: {tool_name} with args {args}")
            tool_result = await client.call(tool_name, **args)

            # Save tool call + result to memory
            memory.append(AIMessage(content=answer))
            memory.append(SystemMessage(content=f"Tool result:\n{tool_result}"))

            # Reinvoke the agent with tool results
            followup = await agent.ainvoke({"messages": normalize_messages(memory)})
            last_ai = [m for m in followup["messages"] if isinstance(m, AIMessage)]
            if not last_ai:
                return str(followup)

            answer = last_ai[-1].content

        except Exception as e:
            return f"❌ Tool execution failed: {e}"

    return answer

# -----------------------------
# Build Agent
# -----------------------------
async def build_agent():
    tools = await client.get_tools()
    llm = ChatOpenAI(
        #model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        model="Qwen/Qwen2.5-7B-Instruct",
        #model="mistralai/Mistral-7B-Instruct-v0.3",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        temperature=0.2,
    )
    agent = create_react_agent(llm, tools)
    return agent, llm

# -----------------------------
# CLI loop
# -----------------------------
async def main():
    print("🤖 B2CC MCP + vLLM Agent")
    agent, llm = await build_agent()
    memory = []

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        target = route_query(user_input)
        print(f"🔎 Router: classified as {target}")

        memory.append(HumanMessage(content=user_input))

        if target == "retriever":
            docs = retriever.invoke(user_input)
            if docs:
                answer = await answer_from_docs(llm, user_input, docs)
            else:
                answer = "I couldn’t find relevant documents for that query."
            print(f"Agent: {answer}")
            memory.append(AIMessage(content=answer))

        else:
            # Add routing hint
            if target == "mysql":
                memory.append(SystemMessage(content=f"User question is about MYSQL. Use the MYSQL MCP tool. Schema:\n{MYSQL_SCHEMA}"))
            elif target == "gitlab":
                memory.append(SystemMessage(content=f"User question is about GitLab. Use the GitLab MCP tool. Schema:\n{GITLAB_SCHEMA}"))

            safe_messages = normalize_messages(memory)
            result = await agent.ainvoke({"messages": safe_messages})
            answer = await handle_tool_calls(agent, client, memory, result)

            print(f"Agent: {answer}")
            memory.append(AIMessage(content=answer))

        # Trim memory
        if len(memory) > 30:
            memory = memory[-30:]

if __name__ == "__main__":
    asyncio.run(main())