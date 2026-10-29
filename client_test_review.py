#!/usr/bin/env python3
"""Live test for depends_on waves, the verify retry loop, brief, and guards."""
import importlib.util
import json
import os
import shutil
import tempfile

os.environ["MODELROUTER_BOOT_PROBE"] = "0"

SERVER = "/home/ak1h/hermes-modelrouter/server.py"
spec = importlib.util.spec_from_file_location("srv", SERVER)
srv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srv)

fails = []


def check(label, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + label + ((" | " + str(detail)) if detail else ""))
    if not cond:
        fails.append(label)


# 1. out-conflict refused before any model call
jobs = [{"id": "a", "role": "code", "prompt": "x", "out": "same.py"},
        {"id": "b", "role": "code", "prompt": "y", "out": "same.py"}]
r = json.loads(srv._plan_impl(json.dumps(jobs), preflight=False, isolate=False))
check("out-conflict refused", r.get("ok") is False and r.get("out_conflicts"),
      r.get("error"))

# 2. unknown depends_on refused
jobs = [{"id": "a", "role": "code", "prompt": "x", "out": "a.py"},
        {"id": "b", "role": "code", "prompt": "y", "out": "b.py",
         "depends_on": ["nope"]}]
r = json.loads(srv._plan_impl(json.dumps(jobs), preflight=False))
check("unknown depends_on refused", r.get("ok") is False and r.get("unknown"),
      r.get("error"))

# 3. real producer -> consumer wave with verify
root = tempfile.mkdtemp(prefix="mr-dep-")
repo = os.path.join(root, "repo")
os.makedirs(repo)
srv.git_ensure_repo(repo)
jobs = [
    {"id": "mod", "role": "code",
     "prompt": "Write mathx.py containing only:\n\ndef triple(n):\n    return n * 3\n",
     "out": "mathx.py", "isolate": True, "repo": repo,
     "verify": "python3 -c 'import mathx; assert mathx.triple(4) == 12'"},
    {"id": "use", "role": "code",
     "prompt": "Write use.py that imports triple from mathx and prints triple(5).",
     "out": "use.py", "isolate": True, "repo": repo, "depends_on": ["mod"],
     "verify": "python3 use.py | grep -qx 15"},
]
r = json.loads(srv._plan_impl(json.dumps(jobs), parallel=2, preflight=False,
                              repo=repo, isolate=True, brief="Python 3, stdlib only."))
res = {x["id"]: x for x in r.get("results") or []}
check("both jobs ok", r.get("ok") is True, r.get("failed"))
check("producer verify passed", (res.get("mod") or {}).get("verify", {}).get("ok") is True,
      (res.get("mod") or {}).get("verify"))
check("consumer verify passed in wave 2",
      (res.get("use") or {}).get("verify", {}).get("ok") is True,
      (res.get("use") or {}).get("verify"))

# 4. verify retry fires on a failing artifact
root2 = tempfile.mkdtemp(prefix="mr-retry-")
repo2 = os.path.join(root2, "repo")
os.makedirs(repo2)
srv.git_ensure_repo(repo2)
jobs = [{"id": "bad", "role": "code",
         "prompt": "Write bad.py containing exactly: print(1)\n",
         "out": "bad.py", "isolate": True, "repo": repo2,
         "verify": "python3 -c 'import sys; sys.exit(1)'", "max_retries": 1}]
r = json.loads(srv._plan_impl(json.dumps(jobs), preflight=False, repo=repo2,
                              isolate=True))
rec = (r.get("results") or [{}])[0]
check("verify failure marks job failed", rec.get("ok") is False, rec.get("error"))
check("retried once", rec.get("attempts") == 2, rec.get("attempts"))

# 5. race fallback: dead model alongside a live one
r = srv.run_with_fallback(["definitely/not-a-model", "DSapi/deepseek-v4.1-flash"],
                          "Reply with the single word OK.", race=2,
                          max_tokens=8, timeout=90)
check("race returns live answer", bool(r.get("ok")), r.get("model"))
check("race recorded attempts", bool(r.get("tried")), r.get("tried"))

# 6. probe keeps unknown
p = json.loads(srv.probe("definitely/not-a-model,DSapi/deepseek-v4.1-flash",
                         parallel=2, timeout=30))
check("probe reports unknown", "unknown" in p, list(p.keys()))

for d in (root, root2):
    shutil.rmtree(d, ignore_errors=True)
print("\nFAILURES: %s" % fails if fails else "\nREVIEWTEST OK")
raise SystemExit(1 if fails else 0)