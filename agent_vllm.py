#!/usr/bin/env python
import asyncio

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
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
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
# Helper: turn docs into human answers
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
# Build Agent with vLLM backend
# -----------------------------
async def build_agent():
    tools = await client.get_tools()

    llm = ChatOpenAI(
        model="Qwen/Qwen2.5-7B-Instruct",  # ✅ ensure this matches your vLLM server
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",   # vLLM ignores this
        temperature=0.2,
    )

    agent = create_react_agent(llm, tools)
    return agent, llm


# -----------------------------
# Router (Method 3 + schema hint)
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
# Normalize messages for vLLM
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
# CLI loop
# -----------------------------
async def main():
    print("🤖 B2CC MCP + vLLM Agent")

    agent, llm = await build_agent()
    memory = []  # in-memory only, capped to ~15 turns

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
            # Add a routing hint to the agent
            if target == "mysql":
                hint = f"User question is about MYSQL. Use the MYSQL MCP tool. Here is the schema:\n{MYSQL_SCHEMA}"
                memory.append(SystemMessage(content=hint))
            elif target == "gitlab":
                hint = f"User question is about GitLab. Use the GitLab MCP tool. Here is the schema:\n{GITLAB_SCHEMA}"
                memory.append(SystemMessage(content=hint))

            safe_messages = normalize_messages(memory)
            result = await agent.ainvoke({"messages": safe_messages})

            messages = result["messages"]
            last_ai = [m for m in messages if isinstance(m, AIMessage)]
            if last_ai:
                answer = last_ai[-1].content
            else:
                answer = str(result)

            print(f"Agent: {answer}")
            memory.append(AIMessage(content=answer))

        # ✅ Keep only the last 15 user+assistant turns
        if len(memory) > 30:
            memory = memory[-30:]


if __name__ == "__main__":
    asyncio.run(main())
