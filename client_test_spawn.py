#!/usr/bin/env python3
"""Smoke test for spawn_agents (planner-chosen fan-out)."""
import asyncio
import json
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_HERE = os.path.dirname(os.path.abspath(__file__))
_SERVER = os.environ.get("MODELROUTER_SERVER") or os.path.join(
    os.path.dirname(_HERE), "server.py")
if not os.path.exists(_SERVER):
    _SERVER = os.path.join(_HERE, "server.py")

SERVER_CMD = "uv"
SERVER_ARGS = ["run", "--with", "mcp", "python3", _SERVER]


async def call(session, name, args):
    r = await session.call_tool(name, args)
    txt = r.content[0].text if r.content else ""
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {"_raw": txt[:800]}


async def main():
    params = StdioServerParameters(command=SERVER_CMD, args=SERVER_ARGS)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = sorted(t.name for t in (await session.list_tools()).tools)
            print("TOOLS:", tools)
            assert "spawn_agents" in tools, "spawn_agents missing"

            print("\n--- dry_run, max_agents=1 (planner must pick 1) ---")
            d = await call(session, "spawn_agents", {
                "goal": "Write a single Python function that reverses a string.",
                "max_agents": 1, "min_agents": 1, "dry_run": True,
                "timeout": 120})
            print("ok:", d.get("ok"), "agents:", d.get("agents"),
                  "planner:", d.get("planner"))
            print("plan:", json.dumps(d.get("plan"), ensure_ascii=False)[:500])

            print("\n--- dry_run, max_agents=5 (multi-part goal) ---")
            d = await call(session, "spawn_agents", {
                "goal": "Build a tiny TODO CLI in Python: (1) a storage module "
                        "that reads and writes JSON, (2) a CLI entry point with "
                        "add/list/done commands, (3) a README.",
                "max_agents": 5, "min_agents": 2, "dry_run": True,
                "timeout": 180})
            print("ok:", d.get("ok"), "agents:", d.get("agents"),
                  "requested:", d.get("requested"), "planner:", d.get("planner"))
            for a in d.get("plan") or []:
                print("  ", a["id"], "|", a["role"], "|", a["out"], "|",
                      a["task"][:90].replace("\n", " "))
            print("plan_health:", json.dumps(d.get("plan_health"))[:200])

            print("\n--- bad input: empty goal ---")
            d = await call(session, "spawn_agents", {"goal": "   "})
            print("ok:", d.get("ok"), "error:", d.get("error"))

            print("\n--- clamp: max_agents=99 -> 12, min_agents=50 -> <=max ---")
            d = await call(session, "spawn_agents", {
                "goal": "Say hello.", "max_agents": 99, "min_agents": 50,
                "dry_run": True, "timeout": 120})
            print("ok:", d.get("ok"), "agents:", d.get("agents"),
                  "requested:", d.get("requested"))


asyncio.run(main())