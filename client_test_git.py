#!/usr/bin/env python3
"""Git isolation test for modelrouter."""
import asyncio
import json
import os
import shutil
import subprocess

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_CMD = "uv"
SERVER_ARGS = ["run", "--with", "mcp", "python3",
               "/home/ak1h/.hermes/mcp/modelrouter/server.py"]
REPO = "/home/ak1h/.hermes/cache/scratch/git-iso-test"


def seed():
    shutil.rmtree(REPO, ignore_errors=True)
    os.makedirs(REPO)
    open(os.path.join(REPO, "README.md"), "w").write("# base\n")
    open(os.path.join(REPO, "keep.py"), "w").write("x = 1\n")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=REPO, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                    "add", "-A"], cwd=REPO, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "-m", "seed"], cwd=REPO, check=True)


async def call(session, name, args):
    r = await session.call_tool(name, args)
    txt = r.content[0].text if r.content else ""
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {"_raw": txt}


async def main():
    seed()
    params = StdioServerParameters(command=SERVER_CMD, args=SERVER_ARGS)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("--- git_status before ---")
            d = await call(session, "git_status", {"repo": REPO})
            print("is_repo:", d.get("is_repo"), "branch:", d.get("branch"),
                  "dirty:", d.get("dirty"))

            print("\n--- plan isolate=True (2 jobs clash on README.md) ---")
            jobs = json.dumps([
                {"id": "alpha", "role": "fast", "repo": REPO, "isolate": True,
                 "out": "README.md", "max_tokens": 32, "timeout": 120,
                 "prompt": "Output exactly this one line and nothing else: "
                           "# written by alpha"},
                {"id": "beta", "role": "fast", "repo": REPO, "isolate": True,
                 "out": "README.md", "max_tokens": 32, "timeout": 120,
                 "prompt": "Output exactly this one line and nothing else: "
                           "# written by beta"},
                {"id": "gamma", "role": "fast", "repo": REPO, "isolate": True,
                 "out": "extra.py", "max_tokens": 32, "timeout": 120,
                 "prompt": "Output exactly this one line and nothing else: "
                           "y = 2"},
            ])
            d = await call(session, "plan",
                           {"jobs_json": jobs, "parallel": 3, "preflight": True})
            iso = d.get("isolation", {})
            print("ok:", d.get("ok"), "failed:", d.get("failed"))
            print("isolation enabled:", iso.get("enabled"),
                  "errors:", iso.get("errors"))
            print("branches:", {k: v.get("branch") for k, v in (iso.get("branches") or {}).items()})
            print("repos:", iso.get("repos"))
            for r in d.get("results", []):
                print(" ", r["id"], "model=", r.get("model"),
                      "branch=", r.get("branch"),
                      "files=", r.get("files_changed"),
                      "commit=", (r.get("commit") or {}).get("committed"))
            print("overlaps:", d.get("overlaps"))

            print("\n--- git_status after ---")
            d2 = await call(session, "git_status", {"repo": REPO})
            print("branch:", d2.get("branch"), "dirty:", d2.get("dirty"))
            print("worktrees:", [w.get("branch") for w in d2.get("worktrees", [])])

            print("\n--- agent_merge alpha (clean) ---")
            b = {k: v.get("branch") for k, v in (iso.get("branches") or {}).items()}
            if b.get("alpha"):
                d3 = await call(session, "agent_merge",
                                {"repo": REPO, "branches": b["alpha"]})
                print("merged:", d3.get("merged"), "failed:", d3.get("failed"))
                readme = open(os.path.join(REPO, "README.md")).read().strip()
                print("README now:", repr(readme))

            print("\n--- agent_merge beta + gamma (beta should conflict) ---")
            names = [b[k] for k in ("beta", "gamma") if b.get(k)]
            d4 = await call(session, "agent_merge",
                            {"repo": REPO, "branches": ",".join(names)})
            print("merged:", d4.get("merged"))
            for f in d4.get("failed", []):
                print("  FAILED", f.get("branch"), "conflicts:", f.get("conflicts"),
                      "err:", f.get("error"))
            readme = open(os.path.join(REPO, "README.md")).read().strip()
            print("README after:", repr(readme))
            extra = os.path.exists(os.path.join(REPO, "extra.py"))
            print("extra.py merged:", extra)

            print("\n--- git_status final ---")
            d5 = await call(session, "git_status", {"repo": REPO})
            print("dirty:", d5.get("dirty"))


asyncio.run(main())
