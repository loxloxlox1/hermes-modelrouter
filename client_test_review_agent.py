import importlib.util, json, os, shutil, tempfile
os.environ["MODELROUTER_BOOT_PROBE"] = "0"
spec = importlib.util.spec_from_file_location("srv", "/home/ak1h/hermes-modelrouter/server.py")
srv = importlib.util.module_from_spec(spec); spec.loader.exec_module(srv)

fails = []
def ck(l, c, d=""):
    print(("ok   " if c else "FAIL ") + l + ((" | " + str(d)) if d else ""))
    if not c: fails.append(l)

# stub: writer answers BAD first then GOOD; reviewer rejects the first only
STATE = {"writer": 0, "rev": []}
def fake(models, prompt, **kw):
    if "You review ONE artifact" in prompt:
        STATE["rev"].append(prompt)
        # reject while the artifact still says BROKEN
        bad = "BROKEN" in prompt
        body = {"ok": not bad,
                "blocking": ["function is not implemented"] if bad else [],
                "notes": "stub" if bad else ""}
        return {"ok": True, "model": "stub/reviewer", "text": json.dumps(body), "secs": 0.1, "tried": []}
    STATE["writer"] += 1
    txt = "print('BROKEN')\n" if STATE["writer"] == 1 else "print('FIXED')\n"
    return {"ok": True, "model": "stub/writer", "text": txt, "secs": 0.1, "tried": []}
srv.run_with_fallback = fake
srv.resolve = lambda x: [x] if isinstance(x, str) and "/" in x else (["stub/writer"] if x == "code" else ["stub/reviewer"])

root = tempfile.mkdtemp(prefix="mrrv-")
jobs = [{"id": "a", "model": "stub/writer", "prompt": "do it", "out": "a.py",
         "reviewer": "stub/reviewer", "review_rounds": 1}]
r = json.loads(srv._plan_impl(json.dumps(jobs), preflight=False, repo="", isolate=False,
                              review=True, review_rounds=1))
res = (r.get("results") or [{}])[0]
ck("job ok after redo", res.get("ok") is True, res.get("error"))
ck("two attempts", res.get("attempts") == 2, res.get("attempts"))
ck("reviewer re-checked the redo", len(STATE["rev"]) == 2, len(STATE["rev"]))
ck("both reviews recorded", len(res.get("reviews") or []) == 2, res.get("reviews"))
ck("final text is the fixed one", "FIXED" in (res.get("chars") and open(res["out"]).read() or ""), res.get("out"))

# review disabled -> no reviewer call
STATE.update(writer=0, rev=[])
jobs = [{"id": "b", "model": "stub/writer", "prompt": "do it", "out": "b.py"}]
r = json.loads(srv._plan_impl(json.dumps(jobs), preflight=False, review=False))
ck("review off: accepted, no reviewer", (r["results"][0]["ok"]) and not STATE["rev"], len(STATE["rev"]))

# reviewer unreachable -> accept, don't fail the job
STATE.update(writer=0, rev=[])
def dead_rev(models, prompt, **kw):
    if "You review ONE artifact" in prompt:
        return {"ok": False, "model": "stub/reviewer", "error": "HTTP 500"}
    STATE["writer"] += 1
    return {"ok": True, "model": "stub/writer", "text": "x = 1\n", "secs": 0.1, "tried": []}
srv.run_with_fallback = dead_rev
jobs = [{"id": "c", "model": "stub/writer", "prompt": "do it", "out": "c.py",
         "reviewer": "stub/reviewer"}]
r = json.loads(srv._plan_impl(json.dumps(jobs), preflight=False, review=True))
rec = r["results"][0]
ck("unreachable reviewer accepts", rec["ok"] is True and rec["review"].get("skipped") is True, rec.get("review"))

# reviewer that returns prose -> accept, flagged skipped
STATE.update(writer=0, rev=[])
def prose_rev(models, prompt, **kw):
    if "You review ONE artifact" in prompt:
        return {"ok": True, "model": "stub/reviewer", "text": "Looks fine to me.", "secs": 0.1, "tried": []}
    return {"ok": True, "model": "stub/writer", "text": "y = 2\n", "secs": 0.1, "tried": []}
srv.run_with_fallback = prose_rev
jobs = [{"id": "d", "model": "stub/writer", "prompt": "do it", "out": "d.py", "reviewer": "stub/reviewer"}]
r = json.loads(srv._plan_impl(json.dumps(jobs), preflight=False, review=True))
ck("prose reviewer accepts", r["results"][0]["ok"] is True, r["results"][0].get("review"))

# exhaustion: writer never fixes -> job fails with blocking list
STATE.update(writer=0, rev=[])
def stub2(models, prompt, **kw):
    if "You review ONE artifact" in prompt:
        return {"ok": True, "model": "stub/reviewer",
                "text": json.dumps({"ok": False, "blocking": ["still broken"], "notes": "n"}),
                "secs": 0.1, "tried": []}
    return {"ok": True, "model": "stub/writer", "text": "bad\n", "secs": 0.1, "tried": []}
srv.run_with_fallback = stub2
jobs = [{"id": "e", "model": "stub/writer", "prompt": "do it", "out": "e.py",
         "reviewer": "stub/reviewer"}]
r = json.loads(srv._plan_impl(json.dumps(jobs), preflight=False, review=True, review_rounds=1))
rec = r["results"][0]
ck("exhausted review fails the job", rec["ok"] is False, rec.get("error"))
ck("error names the blocking defect", "still broken" in (rec.get("error") or ""), rec.get("error"))
ck("withheld from downstream", rec.get("_text", "MISSING") == "MISSING" or not rec.get("_text"), rec.get("_text"))

# verify + review combine: verify retries then review accepts
STATE.update(writer=0, rev=[])
shutil.rmtree(root, ignore_errors=True)
print("\nFAILURES: %s" % fails if fails else "\nREVIEWTEST OK")
raise SystemExit(1 if fails else 0)
