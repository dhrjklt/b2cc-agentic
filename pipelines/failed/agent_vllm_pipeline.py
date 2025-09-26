"""
title: B2CC MCP + vLLM Agent (Streaming)
author: Dhiraj
description: LangGraph + MCP agent with true token streaming for OpenWebUI
required_open_webui_version: 0.4.3
version: 0.3
licence: MIT
"""

import os
import sys
import asyncio
from typing import List, Union, Generator, Iterator
from logging import getLogger
from queue import Queue
from threading import Thread

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel

# --------------------------------------------------------------------
# Import agent_vllm.py from project root
# --------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from agent_vllm import (
        build_agent,
        retriever,
        route_query,
        answer_from_docs,  # kept for non-stream fallback
        normalize_messages,
        MYSQL_SCHEMA,
        GITLAB_SCHEMA,
    )
except ImportError as e:
    raise ImportError(
        f"❌ Could not import agent_vllm.py. Make sure it is in {ROOT_DIR}. Error: {e}"
    )

# --------------------------------------------------------------------
# Logger
# --------------------------------------------------------------------
logger = getLogger(__name__)
logger.setLevel("DEBUG")


class Pipeline:
    """
    B2CC MCP + vLLM Agent pipeline for OpenWebUI with token streaming.
    """

    class Valves(BaseModel):
        pass  # no valves yet

    def __init__(self):
        self.name = "B2CC Agent (Streaming)"
        self.valves = self.Valves()
        self.agent = None
        self.llm = None
        self.memory: List = []

    async def on_startup(self):
        logger.debug(f"on_startup:{self.name}")
        if self.agent is None or self.llm is None:
            self.agent, self.llm = await build_agent()

    async def on_shutdown(self):
        logger.debug(f"on_shutdown:{self.name}")

    # -----------------------------
    # Internal: stream retriever answer tokens
    # -----------------------------
    async def _stream_retriever_answer(self, user_message: str):
        """
        Streams tokens by calling llm.astream(...) on a contextual prompt
        constructed from the top retrieved docs. This avoids waiting for the
        full answer before yielding.
        """
        docs = retriever.invoke(user_message)
        if not docs:
            # nothing to stream; yield a single line
            yield "I couldn’t find relevant documents for that query.\n"
            return

        # Build prompt inline (so we can stream)
        context = "\n\n".join(d.page_content for d in docs[:3])
        prompt = f"""
You are a helpful assistant.
The user asked: {user_message}

Here are the top retrieved documents:

{context}

Please provide a clear, human-like answer using these docs as context.
If the docs do not fully answer, politely say so.
""".strip()

        # Stream tokens from the LLM
        async for chunk in self.llm.astream([HumanMessage(content=prompt)]):
            # chunk is an AIMessageChunk or similar; extract text safely
            text = getattr(chunk, "content", None)
            if text:
                yield text

    # -----------------------------
    # Internal: stream agent tokens via astream_events
    # -----------------------------
    async def _stream_agent_answer(self, safe_messages: List[dict]):
        """
        Streams tokens from the LangGraph agent using astream_events.
        We listen for chat-model streaming events and yield the text deltas.
        """
        # IMPORTANT: event names/shape may vary by langchain/langgraph version.
        # The v2 event API commonly emits "on_chat_model_stream" for token deltas.
        async for event in self.agent.astream_events({"messages": safe_messages}, version="v2"):
            ev = event.get("event")
            data = event.get("data", {})

            # Token-by-token deltas from the underlying chat model
            # Common shapes: data.get("chunk").content OR data.get("delta")
            if ev in ("on_chat_model_stream", "on_llm_stream"):
                chunk = data.get("chunk")
                if chunk:
                    text = getattr(chunk, "content", None)
                    if text:
                        yield text
                else:
                    # Some versions expose delta directly
                    delta = data.get("delta")
                    if isinstance(delta, str) and delta:
                        yield delta

        # When the stream ends, LangGraph has finished all nodes/tools

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        """
        Main pipeline entrypoint for OpenWebUI.

        If body["stream"] is True, returns a generator that yields tokens.
        Otherwise returns the full text once computed.
        """

        async def run_full_answer() -> str:
            """
            Non-streaming path: compute the full answer and return it.
            """
            self.memory.append(HumanMessage(content=user_message))
            target = route_query(user_message)
            logger.debug(f"Router classified as: {target}")

            if target == "retriever":
                # Non-stream fallback uses helper
                docs = retriever.invoke(user_message)
                if docs:
                    answer = await answer_from_docs(self.llm, user_message, docs)
                else:
                    answer = "I couldn’t find relevant documents for that query."
                self.memory.append(AIMessage(content=answer))
                return answer
            else:
                # Add schema hints
                if target == "mysql":
                    self.memory.append(SystemMessage(content=f"Use MYSQL MCP. Schema:\n{MYSQL_SCHEMA}"))
                elif target == "gitlab":
                    self.memory.append(SystemMessage(content=f"Use GitLab MCP. Schema:\n{GITLAB_SCHEMA}"))

                safe_messages = normalize_messages(self.memory)
                result = await self.agent.ainvoke({"messages": safe_messages})
                msgs = result["messages"]
                last_ai = [m for m in msgs if isinstance(m, AIMessage)]
                if last_ai:
                    answer = last_ai[-1].content
                else:
                    answer = str(result)
                self.memory.append(AIMessage(content=answer))
                return answer

        async def run_streaming(queue: Queue):
            """
            Streaming path: push incremental tokens into a thread-safe Queue.
            """
            # Append the new user message into memory for routing + hints
            self.memory.append(HumanMessage(content=user_message))
            target = route_query(user_message)
            logger.debug(f"Router classified as: {target}")

            try:
                if target == "retriever":
                    # Stream tokens directly from llm.astream()
                    async for token in self._stream_retriever_answer(user_message):
                        queue.put(token)
                    queue.put(None)  # sentinel
                    return

                # MCP / LangGraph path
                if target == "mysql":
                    self.memory.append(SystemMessage(content=f"Use MYSQL MCP. Schema:\n{MYSQL_SCHEMA}"))
                elif target == "gitlab":
                    self.memory.append(SystemMessage(content=f"Use GitLab MCP. Schema:\n{GITLAB_SCHEMA}"))

                safe_messages = normalize_messages(self.memory)

                # Token streaming from the agent via events
                async for token in self._stream_agent_answer(safe_messages):
                    queue.put(token)

                # Done streaming; also capture final answer for memory
                # (Optional) You can do a final ainvoke to materialize the final message
                result = await self.agent.ainvoke({"messages": safe_messages})
                msgs = result["messages"]
                last_ai = [m for m in msgs if isinstance(m, AIMessage)]
                if last_ai:
                    self.memory.append(last_ai[-1])

            except Exception as e:
                queue.put(f"\n[streaming error] {e}\n")
            finally:
                queue.put(None)  # sentinel

        # -----------------------------
        # Non-streaming: return full text
        # -----------------------------
        if not body.get("stream", True):
            return asyncio.run(run_full_answer())

        # -----------------------------
        # Streaming: bridge async → sync via a queue + background thread
        # -----------------------------
        q: Queue = Queue()

        def _runner():
            try:
                asyncio.run(run_streaming(q))
            except Exception as e:
                q.put(f"\n[stream start error] {e}\n")
                q.put(None)

        Thread(target=_runner, daemon=True).start()

        def generator():
            while True:
                item = q.get()
                if item is None:
                    break
                # Yield raw token text; OpenWebUI will append to the chat stream
                yield item
            # Ensure there is a trailing newline to finalize the message visually
            yield "\n"

        return generator()
