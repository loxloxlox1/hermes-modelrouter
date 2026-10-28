#!/usr/bin/env python3
"""Smoke test for the modelrouter MCP server."""
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SERVER = os.environ.get("MODELROUTER_SERVER") or os.path.join(
    os.path.dirname(_HERE), "server.py")
if not os.path.exists(_SERVER):
    _SERVER = os.path.join(_HERE, "server.py")

SERVER_CMD = "uv"
SERVER_ARGS = ["run", "--with", "mcp", "python3",
               _SERVER]


async def main():
    params = StdioServerParameters(command=SERVER_CMD, args=SERVER_ARGS)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("TOOLS:", sorted(t.name for t in tools.tools))

            r = await session.call_tool("list_models", {})
            d = json.loads(r.content[0].text)
            print("PROVIDERS:", {k: v["count"] for k, v in d.items()})

            r = await session.call_tool("list_roles", {})
            d = json.loads(r.content[0].text)
            print("ROLES:", {k: [x["model"] for x in v] for k, v in d["roles"].items()})
            print("UNASSIGNED:", d["unassigned_roles"])

            r = await session.call_tool("probe", {"models": "fast", "parallel": 4, "timeout": 60})
            d = json.loads(r.content[0].text)
            print("PROBE alive:", d["alive"])
            print("PROBE dead:", [(x["model"], x["error"]) for x in d["dead"]])

            r = await session.call_tool("ask", {"prompt": "Say OK and nothing else.",
                                                "models": "fast", "max_tokens": 16,
                                                "timeout": 90})
            d = json.loads(r.content[0].text)
            print("ASK ok=%s model=%s text=%r" % (d.get("ok"), d.get("model"),
                                                  (d.get("text") or "")[:60]))

            r = await session.call_tool("auto_route", {"task": "Write a Python function that reverses a string.",
                                                       "models": "fast", "max_tokens": 128,
                                                       "timeout": 120})
            d = json.loads(r.content[0].text)
            print("AUTO_ROUTE role=%s ok=%s model=%s text=%r" % (
                d.get("role"), d.get("ok"), d.get("model"), (d.get("text") or "")[:80]))

asyncio.run(main())