#!/usr/bin/env python3
"""Smoke test for modelrouter preflight health."""
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


async def call(session, name, args):
    r = await session.call_tool(name, args)
    txt = r.content[0].text if r.content else ""
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {"_raw": txt}


async def main():
    params = StdioServerParameters(command=SERVER_CMD, args=SERVER_ARGS)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("TOOLS:", sorted(t.name for t in tools.tools))

            print("\n--- health_check(force) on code role ---")
            d = await call(session, "health_check",
                           {"models": "code", "parallel": 4, "timeout": 60,
                            "force": True})
            print("alive:", d.get("alive"))
            print("dead:", [(x["model"], x["error"]) for x in d.get("dead", [])])
            print("counts:", d.get("counts"), "checked_at:", d.get("checked_at"))

            print("\n--- health_check cached ---")
            d2 = await call(session, "health_check", {"models": "code"})
            print("cached:", d2.get("cached"), "alive:", d2.get("alive"))

            print("\n--- ask with preflight ---")
            d3 = await call(session, "ask",
                            {"prompt": "Say OK and nothing else.", "models": "code",
                             "max_tokens": 16, "timeout": 120, "preflight": True})
            print("ok:", d3.get("ok"), "model:", d3.get("model"),
                  "text:", repr((d3.get("text") or "")[:40]))
            print("health:", d3.get("health"))

            print("\n--- ask preflight=false ---")
            d4 = await call(session, "ask",
                            {"prompt": "Say OK.", "models": "code",
                             "max_tokens": 16, "timeout": 120, "preflight": False})
            print("health:", d4.get("health"),
                  "tried:", [t["model"] for t in d4.get("tried", [])])

            print("\n--- plan with preflight ---")
            jobs = json.dumps([{"id": "j1", "role": "fast", "prompt": "Say A.",
                                "max_tokens": 8, "timeout": 120,
                                "out": "/tmp/modelrouter_j1.txt"}])
            d5 = await call(session, "plan", {"jobs_json": jobs, "preflight": True})
            print("ok:", d5.get("ok"), "failed:", d5.get("failed"))
            print("health:", d5.get("health"))
            print("results:", [(r["id"], r["model"], r["chars"])
                               for r in d5.get("results", [])])


asyncio.run(main())
