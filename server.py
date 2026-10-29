#!/usr/bin/env python3
"""modelrouter — MCP server exposing Hermes' models as a role-routed LLM pool.

Reads model inventory from ~/.hermes/config.yaml and role->candidates from
~/.hermes/agent-fanout/routing.yaml.

Tools:
  list_models   all models Hermes knows (from config.yaml)
  list_roles    role -> ordered candidates
  set_role      assign explicit model list to a role
  auto_route    classify task with an LLM, then run it on that role's models
  ask           one prompt, explicit model or role, with fallback
  plan          parallel multi-job fanout, writes artifacts to disk
  spawn_agents  a planner model chooses the agent count and each agent's task;
                a reviewer model then checks every artifact
  probe         health-check models with a 1-token ping
  health_check  cached pre-flight: which models answer right now, with TTL
  git_status    git repo state: branch, head, dirty files, worktrees
  git_init      make a path a git repo with one commit (worktrees need HEAD)
  agent_merge   merge the agent branches the main model picked; conflicts
                come back aborted with the file list, never force-resolved
  pool_status   which models the startup probe kept, which it dropped
  pool_refresh  re-probe the pool on demand

Every run tool (ask/plan/auto_route) pre-flights its candidates by default and
returns the alive/dead report under "health", so the agent sees model status
before trusting any result. Cache TTL: MODELROUTER_HEALTH_TTL (default 600s).

plan(isolate=True, repo=PATH) gives each job a private git worktree and branch,
commits it, and reports per-branch changed files plus "overlaps" — files two
agents both touched. Nothing is merged automatically; the main model calls
agent_merge with the branches it wants.

On start the server probes every configured model once and records the verdict
in ~/.hermes/agent-fanout/model-pool.json. Dead models are pruned out of role
lookups, so a broken model costs one startup ping instead of a request timeout
on every job. Disable with MODELROUTER_BOOT_PROBE=0; tune the ping timeout with
MODELROUTER_POOL_TIMEOUT (default 45s).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from mcp.server.mcpserver import MCPServer as FastMCP

HOME = os.path.expanduser("~/.hermes")
ENV_FILE = os.path.join(HOME, ".env")
CONFIG = os.path.join(HOME, "config.yaml")
ROUTING = os.path.join(HOME, "agent-fanout", "routing.yaml")

ROLES = ["code", "review", "reason", "design", "research", "fast"]

mcp = FastMCP("modelrouter")


def load_env():
    if not os.path.exists(ENV_FILE):
        return
    for line in open(ENV_FILE, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def key_env_from_url(base):
    """HERMES_CUSTOM_<host>_<port>_API_KEY, the name Hermes uses for a URL."""
    try:
        u = urllib.parse.urlparse(base)
    except ValueError:
        return None
    if not u.hostname:
        return None
    host = re.sub(r"[^0-9A-Za-z]+", "_", u.hostname).strip("_")
    # Providers that use the default port (https://api.foo.com/v1) have no
    # u.port, and Hermes names their key without one.
    if u.port:
        return f"HERMES_CUSTOM_{host}_{u.port}_API_KEY".upper()
    return f"HERMES_CUSTOM_{host}_API_KEY".upper()


def creds(base=None, key_env=None):
    """Resolve (base_url, api_key). An explicit key_env wins for its provider."""
    load_env()
    base = base or os.environ.get("AGENT_FANOUT_BASE") or "https://ai-router.l.cd/v1"
    if key_env:
        key = os.environ.get(key_env)
        if key:
            return base.rstrip("/"), key
    # Local providers in config.yaml often declare no key_env: Hermes derives
    # theirs from the URL. Mirror that, or every local model probes unauthenticated
    # and lands in the dead set.
    derived = key_env_from_url(base)
    if derived:
        key = os.environ.get(derived)
        if key:
            return base.rstrip("/"), key
    key = (os.environ.get("AGENT_FANOUT_KEY")
           or os.environ.get("HERMES_CUSTOM_OSNV_API_KEY")
           or os.environ.get("CUSTOM_PROVIDER_OSNV_KEY")
           or os.environ.get("HERMES_CUSTOM_AI_ROUTER_L_CD_API_KEY"))
    if not key:
        raise RuntimeError("no API key: set AGENT_FANOUT_KEY or HERMES_CUSTOM_OSNV_API_KEY in ~/.hermes/.env")
    return base.rstrip("/"), key


def http_json(url, key, payload=None, timeout=120):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(model, prompt, max_tokens=4096, timeout=240, system=None, temperature=None, base=None):
    key_env = None
    if base is None:
        prov = provider_for(model)
        if prov:
            base = prov["base_url"]
            key_env = prov["key_env"]
    try:
        b, key = creds(base, key_env)
    except RuntimeError as e:
        return {"ok": False, "model": model, "error": f"no credentials: {e}", "secs": 0.0}
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens, "stream": False}
    if temperature is not None:
        body["temperature"] = temperature
    t0 = time.time()
    try:
        r = http_json(b + "/chat/completions", key, body, timeout=timeout)
        txt = (r.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return {"ok": True, "model": model, "text": txt,
                "secs": round(time.time() - t0, 1), "usage": r.get("usage") or {}}
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()[:300]
        except Exception:
            pass
        return {"ok": False, "model": model, "error": f"HTTP {e.code}",
                "detail": detail, "secs": round(time.time() - t0, 1)}
    except Exception as e:
        return {"ok": False, "model": model, "error": f"{type(e).__name__}: {e}",
                "secs": round(time.time() - t0, 1)}


def config_models():
    """Parse providers.*.models out of config.yaml without a YAML dep."""
    if not os.path.exists(CONFIG):
        return {}
    out, prov, in_models = {}, None, False
    for raw in open(CONFIG, encoding="utf-8"):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        s = line.strip()
        if indent == 2 and s.endswith(":") and s not in ("providers:", "model:", "agent:"):
            prov = s[:-1]
            out.setdefault(prov, {"base_url": "", "key_env": None, "models": []})
            in_models = False
        elif prov and indent == 4 and s.startswith("base_url:"):
            out[prov]["base_url"] = s.split(":", 1)[1].strip().strip('"').strip("'")
        elif prov and indent == 4 and s.startswith("key_env:"):
            out[prov]["key_env"] = s.split(":", 1)[1].strip().strip('"').strip("'")
        elif prov and indent == 4 and s == "models:":
            in_models = True
        elif prov and in_models and indent >= 6 and s.endswith(": {}"):
            # Strip exactly the ": {}" suffix (4 chars). Slicing 5 truncated
            # every model id by one character and broke provider lookup.
            out[prov]["models"].append(s[: -len(": {}")].strip())
        elif prov and in_models and indent <= 4:
            in_models = False
    return {k: v for k, v in out.items() if v["models"]}


def provider_for(model):
    """Find the provider block that declares `model`.

    Returns {"base_url", "key_env"} or None. Without this every model is sent
    to the default endpoint, so models on another provider fail with 4xx.
    """
    for info in config_models().values():
        if model in info["models"]:
            return {"base_url": info.get("base_url") or None,
                    "key_env": info.get("key_env") or None}
    return None


def parse_routing():
    if not os.path.exists(ROUTING):
        return {}
    out, cur = {}, None
    for raw in open(ROUTING, encoding="utf-8"):
        # Only " #" starts a comment: model ids may contain a bare '#'.
        line = re.split(r"(^|\s)#", raw, 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.rstrip().endswith(":"):
            cur = line.strip()[:-1]
            out[cur] = []
        elif line.strip().startswith("- ") and cur:
            out[cur].append(line.strip()[2:].strip())
        elif ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                out[k.strip()] = [x.strip() for x in v[1:-1].split(",") if x.strip()]
    return out


def write_routing(routing):
    os.makedirs(os.path.dirname(ROUTING), exist_ok=True)
    lines = ["# Role -> ordered model candidates. First alive wins.",
             "# Written by modelrouter MCP. Edit by hand too.", ""]
    for role, models in routing.items():
        lines.append(f"{role}:")
        for m in models:
            lines.append(f"  - {m}")
        lines.append("")
    tmp = ROUTING + ".tmp"
    open(tmp, "w", encoding="utf-8").write("\n".join(lines))
    os.replace(tmp, ROUTING)


def resolve(role_or_models):
    """Resolve a role / explicit model / fallback chain.

    Role lookups and comma chains are pruned against the persistent pool so a
    model that failed its startup probe is never tried. A single explicit model
    name is returned as-is: the caller asked for that one, so it should fail
    loudly rather than be silently swapped.
    """
    if isinstance(role_or_models, list):
        return pool_prune([str(x) for x in role_or_models])
    s = str(role_or_models).strip()
    routing = parse_routing()
    if s in routing:
        return pool_prune(routing[s])
    if "," in s:
        return pool_prune([x.strip() for x in s.split(",") if x.strip()])
    return [s]


def run_with_fallback(models, prompt, race=1, **kw):
    """Try each model until one returns non-empty text.

    race: how many candidates to launch at once. A dead model otherwise costs a
    full timeout before the next one is tried. First non-empty answer wins.
    """
    models = [m for m in models if m]
    tried = []

    def attempt(m):
        r = chat(m, prompt, **kw)
        if r.get("ok") and not (r.get("text") or "").strip():
            r = dict(r, ok=False, error="empty body (HTTP 200)")
        return r

    n = max(1, min(int(race or 1), len(models) or 1))
    if n > 1 and len(models) > 1:
        head, rest = models[:n], models[n:]
        ex = ThreadPoolExecutor(max_workers=n)
        try:
            futs = {ex.submit(attempt, m): m for m in head}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"ok": False, "model": futs[fut],
                         "error": "%s: %s" % (type(e).__name__, e)}
                tried.append({"model": r.get("model"), "ok": r.get("ok", False),
                              "error": r.get("error")})
                if r.get("ok"):
                    r["tried"] = tried
                    return r
        finally:
            # A `with` block would join every loser: a winner arriving in 1s
            # would still wait out a 240s timeout on the other candidate.
            ex.shutdown(wait=False, cancel_futures=True)
        models = rest
    last = None
    for m in models:
        r = attempt(m)
        tried.append({"model": m, "ok": r.get("ok", False), "error": r.get("error")})
        last = r
        if r.get("ok"):
            r["tried"] = tried
            return r
    if last is None:
        return {"ok": False, "error": "no models to try", "tried": tried}
    last = dict(last)
    last["tried"] = tried
    return last


HEALTH_TTL = int(os.environ.get("MODELROUTER_HEALTH_TTL", "600"))
_health_lock = threading.Lock()
_health = {"alive": None, "dead": None, "ts": 0.0, "source": "none"}


# Probing every model at once trips aggregator rate limits, and a throttled
# model then looks dead. Only definitive failures remove a model from the pool;
# throttling/timeouts are reported as "unknown" so the previous verdict stands.
TRANSIENT_ERRORS = ("HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504",
                    "TimeoutError", "URLError")


def _transient(err):
    return bool(err) and any(t in err for t in TRANSIENT_ERRORS)


# The aggregator throttles bursts: sequential probes of the same endpoint are
# fine, a parallel burst is not. Space probe requests per base_url.
PROBE_MIN_GAP = float(os.environ.get("MODELROUTER_PROBE_GAP", "1.5"))
_gap_locks = {}
_gap_last = {}


def _probe_gate(base):
    if not base:
        return
    with _health_lock:
        lock = _gap_locks.setdefault(base, threading.Lock())
    with lock:
        wait = PROBE_MIN_GAP - (time.time() - _gap_last.get(base, 0))
        if wait > 0:
            time.sleep(wait)
        _gap_last[base] = time.time()


def _probe_one(m, timeout, retries=2):
    last = {}
    prov = provider_for(m) or {}
    for attempt in range(retries):
        _probe_gate(prov.get("base_url"))
        r = chat(m, "ping", max_tokens=4, timeout=timeout)
        err = r.get("error")
        if r["ok"] and not r["text"].strip():
            err = "empty body (HTTP 200)"
        last = {"model": m, "secs": r.get("secs"), "error": err}
        if r["ok"] and r["text"].strip():
            last.update({"ok": True, "status": "alive"})
            return last
        if not _transient(err):
            last.update({"ok": False, "status": "dead"})
            return last
        if attempt + 1 < retries:
            time.sleep(1.5 * (attempt + 1))
    last.update({"ok": False, "status": "unknown"})
    return last


def probe_models(models, parallel=3, timeout=60):
    """Ping each model; return (alive, dead).

    `dead` holds only definitive failures. Throttled/timeout models are
    returned under the "unknown" status and must not lose their pool slot.
    """
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        res = list(ex.map(lambda m: _probe_one(m, timeout), models))
    alive = [r["model"] for r in res if r.get("status") == "alive"]
    dead = [r for r in res if r.get("status") == "dead"]
    unknown = [r for r in res if r.get("status") == "unknown"]
    return alive, dead, unknown


def health_snapshot(models, parallel=8, timeout=60, force=False):
    """Cached health. Probing is expensive, so reuse a fresh snapshot.

    Returns (snapshot, from_cache). from_cache is True only when every
    requested model was already covered by an unexpired snapshot.
    """
    with _health_lock:
        fresh = (not force) and _health["alive"] is not None and (time.time() - _health["ts"] < HEALTH_TTL)
        if fresh:
            covered = set(_health["alive"]) | {d["model"] for d in _health["dead"]}
            if set(models) <= covered:
                return _health, True
    alive, dead, unknown = probe_models(models, parallel=parallel, timeout=timeout)
    unknown_names = {u["model"] for u in unknown}
    with _health_lock:
        prev_alive = set(_health["alive"] or []) - set(models)
        # A model that came back unknown was not re-verified: keep whatever
        # verdict it already had instead of dropping it from both buckets.
        prev_dead = [d for d in (_health["dead"] or [])
                     if d["model"] not in models or d["model"] in unknown_names]
        kept_unknown_dead = [{"model": m, "error": "unknown (cached dead)"}
                             for m in unknown_names
                             if m not in {d["model"] for d in dead}
                             and m in {d["model"] for d in (_health["dead"] or [])}]
        _health.update({"alive": sorted(prev_alive | set(alive) | (
                            {m for m in models if m in unknown_names}
                            - {d["model"] for d in prev_dead + kept_unknown_dead})),
                        "dead": prev_dead + dead + kept_unknown_dead,
                        "ts": time.time(), "source": "live"})
        health = os.path.join(HOME, "agent-fanout", "health.json")
        os.makedirs(os.path.dirname(health), exist_ok=True)
        try:
            json.dump({"alive": _health["alive"], "dead": _health["dead"],
                       "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                      open(health, "w"), indent=1)
        except OSError:
            pass
        return _health, False


def filter_alive(models, report=True):
    """Pre-flight: drop dead models from a candidate list, report what was dropped."""
    if not models:
        return models, {"skipped": [], "checked": False}
    h, from_cache = health_snapshot(models)
    dead = {d["model"] for d in h["dead"]}
    alive = [m for m in models if m not in dead]
    info = {"checked": True, "from_cache": from_cache,
            "skipped_dead": [m for m in models if m in dead], "alive": alive,
            "health_ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(h["ts"]))}
    return (alive or models), info


# ------------------------------------------------------------- model pool
# A persistent alive/dead verdict per model, refreshed by a probe at server
# start. Dead models are removed from role lookups, so a broken model costs one
# startup ping instead of a full request timeout on every job.
POOL_TTL = int(os.environ.get("MODELROUTER_POOL_TTL", "3600"))
POOL_PROBE_TIMEOUT = int(os.environ.get("MODELROUTER_POOL_TIMEOUT", "45"))
POOL_FILE = os.path.join(HOME, "agent-fanout", "model-pool.json")

_pool_lock = threading.Lock()
_pool = {"alive": set(), "dead": set(), "ts": 0.0, "ready": False,
         "probing": False, "last_error": None, "probed": 0}


def pool_load():
    """Restore the last verdict so a restart does not re-trust dead models."""
    try:
        d = json.load(open(POOL_FILE, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    with _pool_lock:
        _pool["alive"] = set(d.get("alive") or [])
        _pool["dead"] = set(d.get("dead") or [])
        _pool["ts"] = float(d.get("ts") or 0)
        _pool["ready"] = bool(d.get("ready")) and bool(_pool["alive"] or _pool["dead"])
        _pool["probed"] = int(d.get("probed") or 0)
    return True


def pool_save():
    with _pool_lock:
        snap = {"alive": sorted(_pool["alive"]), "dead": sorted(_pool["dead"]),
                "ts": _pool["ts"], "ready": _pool["ready"],
                "probed": _pool["probed"], "last_error": _pool["last_error"],
                "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    os.makedirs(os.path.dirname(POOL_FILE), exist_ok=True)
    tmp = POOL_FILE + ".tmp"
    try:
        json.dump(snap, open(tmp, "w", encoding="utf-8"), indent=1)
        os.replace(tmp, POOL_FILE)
    except OSError as e:
        with _pool_lock:
            _pool["last_error"] = f"pool save failed: {e}"


def pool_prune(models):
    """Drop models the pool has marked dead. Never returns an empty chain."""
    if not models:
        return models
    with _pool_lock:
        ready = _pool["ready"]
        dead = set(_pool["dead"])
    if not ready or not dead:
        return list(models)
    kept = [m for m in models if m not in dead]
    return kept or list(models)


def pool_probe(parallel=3, timeout=None, reason="startup"):
    """Probe every configured model and rewrite the pool.

    If every model fails the endpoint is down, not the models — the previous
    verdict is kept so one bad startup cannot empty the pool.
    """
    timeout = timeout or POOL_PROBE_TIMEOUT
    with _pool_lock:
        if _pool["probing"]:
            return {"ok": False, "error": "probe already running"}
        _pool["probing"] = True
    t0 = time.time()
    try:
        targets = sorted({m for i in config_models().values() for m in i["models"]})
        if not targets:
            with _pool_lock:
                _pool["last_error"] = "no models configured"
            return {"ok": False, "error": "no models configured"}
        alive, dead, unknown = probe_models(targets, parallel=parallel,
                                             timeout=timeout)
        dead_names = {d["model"] for d in dead}
        unknown_names = {u["model"] for u in unknown}
        if not alive:
            with _pool_lock:
                _pool["last_error"] = "all models failed; pool unchanged"
            pool_save()
            return {"ok": False, "reason": reason, "probed": len(targets),
                    "error": "all models failed; pool unchanged (endpoint down?)"}
        with _pool_lock:
            prev_dead = set(_pool["dead"])
            prev_alive = set(_pool["alive"])
            # Throttled models keep whatever verdict they already had.
            kept_alive = (prev_alive & unknown_names) | set(alive)
            _pool["alive"] = kept_alive - dead_names
            _pool["dead"] = dead_names - kept_alive
            _pool["ts"] = time.time()
            _pool["ready"] = True
            _pool["probed"] = len(targets)
            _pool["last_error"] = None
            now_dead = sorted(dead_names - prev_dead)
            back = sorted(prev_dead - dead_names)
        pool_save()
        return {"ok": True, "reason": reason, "probed": len(targets),
                "alive": len(alive), "dead": len(dead_names),
                "unknown": len(unknown_names),
                "dead_models": sorted(dead_names),
                "unknown_models": sorted(unknown_names),
                "newly_dead": now_dead, "back_alive": back,
                "secs": round(time.time() - t0, 1),
                "pool_file": POOL_FILE}
    finally:
        with _pool_lock:
            _pool["probing"] = False


@mcp.tool()
def pool_status() -> str:
    """Current model pool: which models are usable and which the probe dropped.

    The pool is rebuilt by a probe when the server starts, and dead models are
    excluded from every role lookup.
    """
    routing = parse_routing()
    with _pool_lock:
        snap = {"ready": _pool["ready"], "probing": _pool["probing"],
                "alive": sorted(_pool["alive"]), "dead": sorted(_pool["dead"]),
                "probed": _pool["probed"], "last_error": _pool["last_error"],
                "checked_at": (time.strftime("%Y-%m-%dT%H:%M:%S",
                                             time.localtime(_pool["ts"]))
                               if _pool["ts"] else None),
                "age_secs": round(time.time() - _pool["ts"], 1) if _pool["ts"] else None}
    dead = set(snap["dead"])
    roles = {role: {"active": [m for m in models if m not in dead],
                    "dropped": [m for m in models if m in dead]}
             for role, models in routing.items()}
    snap["roles"] = roles
    snap["pool_file"] = POOL_FILE
    snap["ttl_secs"] = POOL_TTL
    return json.dumps(snap, indent=1, ensure_ascii=False)


@mcp.tool()
def pool_refresh(models: str = "", parallel: int = 3, timeout: int = 45) -> str:
    """Re-probe the pool now instead of waiting for the next server restart.

    models: comma list or role name; empty = every configured model.
    """
    with _pool_lock:
        if _pool["probing"]:
            return json.dumps({"ok": False, "error": "probe already running"})
    if models:
        targets = resolve(models)
    else:
        targets = sorted({m for i in config_models().values() for m in i["models"]})
    if not targets:
        return json.dumps({"ok": False, "error": "no models"})
    t0 = time.time()
    alive, dead, unknown = probe_models(targets, parallel=parallel, timeout=timeout)
    dead_names = {d["model"] for d in dead}
    if not alive:
        return json.dumps({"ok": False, "probed": len(targets),
                           "error": "all models failed; pool unchanged"},
                          ensure_ascii=False)
    with _pool_lock:
        for m in targets:
            _pool["alive"].discard(m)
            _pool["dead"].discard(m)
        _pool["alive"] |= set(alive)
        _pool["dead"] |= dead_names
        _pool["ts"] = time.time()
        _pool["ready"] = True
        _pool["probed"] = len(_pool["alive"]) + len(_pool["dead"])
        _pool["last_error"] = None
    pool_save()
    return json.dumps({"ok": True, "probed": len(targets), "alive": len(alive),
                       "dead": sorted(dead_names),
                       "secs": round(time.time() - t0, 1),
                       "pool_file": POOL_FILE}, indent=1, ensure_ascii=False)


pool_load()


@mcp.tool()
def list_models(grep: str = "", provider: str = "") -> str:
    """List every model Hermes has configured, grouped by provider.

    grep: regex filter on model id. provider: restrict to one provider name.
    """
    cfgs = config_models()
    rx = re.compile(grep, re.I) if grep else None
    res = {}
    for prov, info in cfgs.items():
        if provider and prov != provider:
            continue
        names = [m for m in info["models"] if not rx or rx.search(m)]
        if names:
            res[prov] = {"base_url": info["base_url"], "key_env": info["key_env"],
                         "count": len(names), "models": names}
    return json.dumps(res, indent=1, ensure_ascii=False)


@mcp.tool()
def list_roles() -> str:
    """Show the current role -> ordered model candidates table."""
    routing = parse_routing()
    known = {m for info in config_models().values() for m in info["models"]}
    res = {}
    for role, models in routing.items():
        res[role] = [{"model": m, "in_config": m in known} for m in models]
    return json.dumps({"routing_file": ROUTING, "roles": res,
                       "unassigned_roles": [r for r in ROLES if r not in routing]},
                      indent=1, ensure_ascii=False)


@mcp.tool()
def set_role(role: str, models: str) -> str:
    """Assign an explicit ordered model list to a role.

    role: code / review / reason / design / research / fast (free-form allowed).
    models: comma-separated, highest priority first. First alive wins.
    """
    role = role.strip()
    if not role:
        return json.dumps({"ok": False, "error": "empty role"})
    lst = [x.strip() for x in models.split(",") if x.strip()]
    if not lst:
        return json.dumps({"ok": False, "error": "empty model list"})
    routing = parse_routing()
    routing[role] = lst
    write_routing(routing)
    return json.dumps({"ok": True, "role": role, "models": lst,
                       "routing_file": ROUTING}, indent=1, ensure_ascii=False)


@mcp.tool()
def auto_route(task: str, models: str = "", max_tokens: int = 4096,
               timeout: int = 240, system: str = "", preflight: bool = True) -> str:
    """Classify a task into a role with a cheap LLM, then run it on that role's models.

    models: optional classifier model (default: first of the `fast` role).
    preflight: ping candidates before spending a full request on them.
    """
    routing = parse_routing()
    classifier_models = resolve(models) if models else (routing.get("fast") or [])[:2]
    if not classifier_models:
        return json.dumps({"ok": False, "error": "no classifier model; pass models=..."})
    classify_prompt = (
        "Classify this task into exactly one label: code, review, reason, design, "
        "research, fast.\n"
        "code=write/refactor programs; review=audit existing code or text; "
        "reason=math/logic/analysis/planning; design=UI, layout, visual, copy; "
        "research=find facts, sources, summaries; fast=trivial short answers.\n"
        "Answer with the single lowercase label, nothing else.\n\n"
        f"TASK:\n{task[:4000]}"
    )
    cls = run_with_fallback(classifier_models, classify_prompt, max_tokens=8,
                            timeout=min(timeout, 90), temperature=0)
    if not cls.get("ok"):
        return json.dumps({"ok": False, "stage": "classify",
                           "error": cls.get("error"), "tried": cls.get("tried")},
                          indent=1, ensure_ascii=False)
    raw = cls["text"].strip().lower()
    role = next((r for r in ROLES if r in raw), "reason")
    targets = routing.get(role) or routing.get("reason") or []
    if not targets:
        return json.dumps({"ok": False, "stage": "route", "role": role,
                           "error": "role has no candidates; run set_role"})
    health = {"checked": False}
    if preflight:
        targets, health = filter_alive(targets)
    res = run_with_fallback(targets, task, max_tokens=max_tokens,
                            timeout=timeout, system=system or None)
    res.update({"role": role, "classified_by": cls["model"], "classifier_raw": raw,
                "health": health})
    return json.dumps(res, indent=1, ensure_ascii=False)


@mcp.tool()
def ask(prompt: str, models: str = "reason", max_tokens: int = 4096,
        timeout: int = 240, system: str = "", preflight: bool = True) -> str:
    """Run one prompt against a role or explicit model(s), with fallback.

    models: a role name (code/review/reason/design/research/fast),
            a model id, or a comma-separated fallback chain.
    preflight: ping the candidates first and drop dead ones; the health
               report is returned in "health" so the agent sees it.
    """
    targets = resolve(models)
    if not targets:
        return json.dumps({"ok": False, "error": f"unresolved: {models}"})
    health = {"checked": False}
    if preflight:
        targets, health = filter_alive(targets)
    res = run_with_fallback(targets, prompt, max_tokens=max_tokens,
                            timeout=timeout, system=system or None)
    res["requested"] = models
    res["health"] = health
    return json.dumps(res, indent=1, ensure_ascii=False)


def _plan_impl(jobs_json: str, parallel: int = 4, preflight: bool = True,
               repo: str = "", isolate: bool = False, brief: str = "",
               race: int = 2, review: bool = True, reviewer: str = "",
               review_rounds: int = 1) -> str:
    """Run a fanout plan. Each job writes its output to `out`.

    jobs_json: JSON list. Per job: id, role|model|models, prompt|prompt_file,
               optional files[], system, max_tokens, timeout, out, repo,
               branch, isolate, depends_on[], verify, max_retries.
    repo: default git repo for isolated jobs (a job's own `repo` wins).
    isolate: give each job its own git worktree + branch, so parallel jobs
             never overwrite each other's files. Requires git.
    brief: shared text appended to every job's system prompt -- stack, naming,
           API contract. Parallel-safe, no dependencies, covers most of what a
           shared transcript would and costs one copy per prompt.
    race: how many candidate models to launch at once per job.
    depends_on: ids that must finish first. Jobs run in waves; a job starts
        only after its ancestors are done, their artifacts are copied into its
        worktree and pasted into its prompt. Jobs without depends_on form the
        first wave and run in parallel.
    verify: shell command run in the job's workspace after it writes. Non-zero
        exit feeds the output back to the model and retries, up to max_retries
        (default 2).
    review: after a job passes verify, a separate reviewer model reads the
        artifact and either accepts it or lists blocking defects; a rejected
        job is re-run with those defects in its prompt, up to review_rounds
        times (default 1). Per-job `review: false` opts out; per-job
        `reviewer` picks the model or role.
    preflight: one health sweep over every model the plan touches.
    A job that wrote 0 bytes counts as failed.
    """
    try:
        jobs = json.loads(jobs_json)
    except json.JSONDecodeError as e:
        return json.dumps({"ok": False, "error": f"bad JSON: {e}"})
    if isinstance(jobs, dict):
        jobs = jobs.get("jobs", [])
    if not isinstance(jobs, list) or not jobs:
        return json.dumps({"ok": False, "error": "jobs must be a non-empty list"})

    ids, seen_ids = [], set()
    for i, job in enumerate(jobs):
        jid = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(job.get("id") or f"job-{i + 1}"))
        base, n = jid, 2
        while jid in seen_ids:
            jid = f"{base}-{n}"
            n += 1
        seen_ids.add(jid)
        ids.append(jid)
        if job.get("id") != jid:
            jobs[i] = dict(job, id=jid)

    known = set(ids)
    bad_deps = []
    for jid, job in zip(ids, jobs):
        for d in job.get("depends_on") or []:
            if str(d) not in known:
                bad_deps.append({"job": jid, "depends_on": str(d)})
    if bad_deps:
        return json.dumps({"ok": False,
                           "error": "depends_on refers to an unknown job id",
                           "unknown": bad_deps, "known": ids},
                          indent=1, ensure_ascii=False)

    outs = {}
    for jid, job in zip(ids, jobs):
        o = str(job.get("out") or "")
        if o:
            outs.setdefault(o, []).append(jid)
    out_conflicts = [{"out": k, "jobs": v} for k, v in outs.items() if len(v) > 1]
    if out_conflicts and not isolate:
        return json.dumps({"ok": False,
                           "error": "two jobs write the same out path; give "
                                    "distinct paths or turn on isolate",
                           "out_conflicts": out_conflicts},
                          indent=1, ensure_ascii=False)

    report = {"checked": False}
    reviewer_pool = []
    if review:
        reviewer_pool = resolve(reviewer or "review") or resolve("reason")
    if preflight:
        all_models = []
        for j in jobs:
            ms = (j.get("models") or ([j["model"]] if j.get("model")
                  else resolve(j.get("role", "reason"))))
            all_models += [m for m in ms if m not in all_models]
        all_models += [m for m in reviewer_pool if m not in all_models]
        if all_models:
            report = filter_alive(all_models)[1]

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    git_ok = git_available()
    isolation = {"enabled": False, "run_id": run_id, "git": git_ok,
                 "branches": {}, "errors": []}
    prepared = []

    def wants_isolate(job):
        v = job.get("isolate")
        return bool(v) if v is not None else bool(isolate)

    def _resolve_in(worktree, path, repo_root=""):
        """Resolve a relative path for a job.

        An isolated job runs in a fresh worktree, but the file it was told to
        read may be uncommitted in the real repo. Try the worktree first, then
        the repo root, and only then give up.
        """
        if not path:
            return path
        if os.path.isabs(path):
            return path
        if worktree:
            cand = os.path.join(worktree, path)
            if os.path.exists(cand):
                return cand
        if repo_root:
            cand = os.path.join(repo_root, path)
            if os.path.exists(cand):
                return cand
        return os.path.join(worktree, path) if worktree else path

    isolated_jobs = [j for j in jobs if wants_isolate(j)]
    if isolated_jobs and not git_ok:
        isolation["errors"].append("git is not installed; isolation skipped")
        isolated_jobs = []
    if isolated_jobs:
        repos = {}
        top = os.path.abspath(os.path.expanduser(repo)) if repo else ""
        for job in isolated_jobs:
            r = job.get("repo") or top
            if not r:
                isolation["errors"].append(
                    f"job {job.get('id', '?')}: no repo given; wrote to shared cwd")
                continue
            r = os.path.abspath(os.path.expanduser(r))
            repos.setdefault(r, []).append(job)
        for r, group in repos.items():
            init = git_ensure_repo(r)
            if not init.get("ok"):
                isolation["errors"].append(f"{r}: {init.get('error')}")
                continue
            isolation["enabled"] = True
            isolation.setdefault("repos", {})[init["repo"]] = {
                "base_head": init["head"], "created": init.get("created", False),
                "initial_commit": init.get("initial_commit", False)}
            for job in group:
                idx = next(i for i, j in enumerate(jobs) if j is job)
                jid = job["id"]
                branch = str(job.get("branch") or f"mr/{run_id}/{jid}")
                wt = os.path.join(SCRATCH, "modelrouter-wt", run_id, jid)
                made = git_worktree_add(init["repo"], wt, branch, init["head"])
                if not made.get("ok"):
                    isolation["errors"].append(f"{jid}: {made.get('error')}")
                    continue
                job = dict(job)
                job["_worktree"] = made["worktree"]
                job["_branch"] = made["branch"]
                job["_repo"] = init["repo"]
                isolation["branches"][jid] = {
                    "branch": made["branch"], "repo": init["repo"],
                    "worktree": made["worktree"]}
                prepared.append((job, made["worktree"], idx))

    used = {pair[2] for pair in prepared if len(pair) > 2}
    for idx, job in enumerate(jobs):
        if idx not in used:
            prepared.append((job, None, idx))

    ART_CAP = 6000
    CTX_CAP = 18000
    ctx = {}

    def _read(path):
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            return None

    def _ancestor_block(job):
        parts, total = [], 0
        for d in (job.get("depends_on") or []):
            rec = ctx.get(str(d)) or {}
            body = _read(rec["out"]) if rec.get("out") else None
            if body is None:
                body = rec.get("text") or ""
            if not body.strip():
                continue
            body = body[:ART_CAP]
            parts.append("\n\n--- ARTIFACT FROM %s (%s) ---\n%s"
                         % (d, rec.get("out") or "inline", body))
            total += len(body)
            if total >= CTX_CAP:
                break
        if not parts:
            return ""
        return ("\n\n--- ANCESTOR OUTPUT ---\nThese files were produced by the "
                "jobs you depend on. Match their API and names exactly."
                + "".join(parts))

    def _seed_worktree(job, worktree):
        """Copy ancestors' artifacts into a dependent job's worktree."""
        if not worktree:
            return
        for d in (job.get("depends_on") or []):
            rec = ctx.get(str(d)) or {}
            src = rec.get("out")
            if not src or not os.path.isfile(src):
                continue
            # The ancestor's `out` may be relative ("src/x.py"); copying by
            # basename would land it at the worktree root and the dependent
            # job would then import the wrong path.
            rel = rec.get("rel") or os.path.basename(src)
            dst = os.path.join(worktree, rel)
            try:
                os.makedirs(os.path.dirname(dst) or worktree, exist_ok=True)
                shutil.copyfile(src, dst)
            except OSError:
                pass

    def _verify(job, worktree, fallback_dir):
        cmd = str(job.get("verify") or "").strip()
        if not cmd:
            return {"ok": True, "skipped": True}
        cwd = worktree or fallback_dir or os.getcwd()
        limit = int(job.get("verify_timeout", 300))
        try:
            p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                               text=True, timeout=limit)
        except subprocess.TimeoutExpired:
            return {"ok": False, "cmd": cmd, "error": f"timeout after {limit}s"}
        except OSError as e:
            return {"ok": False, "cmd": cmd, "error": f"could not run: {e}"}
        out = ((p.stdout or "") + (p.stderr or "")).strip()
        return {"ok": p.returncode == 0, "cmd": cmd, "exit_code": p.returncode,
                "output": out[-3000:]}

    ART_REVIEW_CAP = 14000

    def _review(job, worktree, out_path, text, task, pool):
        """One reviewer pass: {ok, blocking, notes, model, skipped}."""
        body = text or ""
        if out_path and os.path.isfile(out_path):
            body = _read(out_path) or body
        if not body.strip():
            return {"ok": False, "model": None, "skipped": False,
                    "blocking": ["artifact is empty"], "notes": "nothing to review"}
        pool = [m for m in (pool or [])]
        writer = job.get("model") or ""
        others = [m for m in pool if m != writer]
        if others:
            pool = others
        if not pool:
            return {"ok": True, "model": None, "skipped": True, "blocking": [],
                    "notes": "no reviewer model available"}
        prompt = (REVIEWER_PROMPT
                  .replace("__PATH__", job.get("out") or "inline")
                  .replace("__TASK__", (task or "")[:4000])
                  .replace("__ARTIFACT__", body[:ART_REVIEW_CAP]))
        r = run_with_fallback(pool, prompt, max_tokens=1200,
                              timeout=min(int(job.get("timeout", 600)), 300),
                              temperature=0, race=1)
        if not r.get("ok"):
            return {"ok": True, "model": r.get("model"), "skipped": True,
                    "blocking": [], "notes": "reviewer unreachable: %s" % r.get("error")}
        parsed = _extract_json(r.get("text"))
        if not isinstance(parsed, dict) or "ok" not in parsed:
            return {"ok": True, "model": r.get("model"), "skipped": True,
                    "blocking": [], "raw": (r.get("text") or "")[:800],
                    "notes": "reviewer did not return JSON; treated as accept"}
        blocking = [str(x) for x in (parsed.get("blocking") or []) if str(x).strip()]
        return {"ok": bool(parsed.get("ok")) and not blocking,
                "model": r.get("model"), "skipped": False,
                "blocking": blocking,
                "notes": str(parsed.get("notes") or "")[:2000]}

    def run(pair):
        job, worktree = pair[0], pair[1]
        jid = job.get("id", "?")
        targets = (job.get("models") or ([job["model"]] if job.get("model")
                   else resolve(job.get("role", "reason"))))
        base = job.get("prompt", "")
        if job.get("prompt_file"):
            try:
                base = open(_resolve_in(worktree, job["prompt_file"],
                                        job.get("_repo") or repo),
                            encoding="utf-8").read()
            except OSError as e:
                return {"id": jid, "ok": False, "error": f"prompt_file: {e}",
                        "branch": job.get("_branch")}
        extra = ""
        for f in job.get("files") or []:
            fp = _resolve_in(worktree, f, job.get("_repo") or repo)
            try:
                extra += ("\n\n--- FILE: %s ---\n" % fp) + open(
                    fp, encoding="utf-8", errors="replace").read()
            except OSError as e:
                return {"id": jid, "ok": False, "error": f"file {f}: {e}",
                        "branch": job.get("_branch")}
        note = ""
        if worktree:
            note = (f"\n\n--- WORKSPACE ---\nEdit files inside {worktree}. "
                    "Write every output file there. Do not touch other paths. "
                    "Put the file content itself in the file -- no commentary.")
        system = job.get("system") or ""
        if brief:
            system = (system + "\n\n--- PROJECT BRIEF ---\n" + brief
                      if system.strip() else brief)
        prompt = base + extra + _ancestor_block(job) + note
        want_review = bool(job.get("review", review)) and bool(reviewer_pool)
        rounds = max(0, min(int(job.get("review_rounds", review_rounds) or 0), 3))
        my_pool = resolve(job["reviewer"]) if job.get("reviewer") else reviewer_pool
        retries = int(job.get("max_retries", 2)) if job.get("verify") else 0
        total = max(retries, 0) + (rounds if want_review else 0) + 1
        rec = {"id": jid, "role": job.get("role", ""), "ok": False,
               "model": None, "secs": None, "error": None, "tried": None,
               "_text": "", "reviews": []}
        ver = {"ok": True, "skipped": True}
        rev = {"ok": True, "skipped": True}
        attempts = 0
        feedback = ""
        while attempts < total:
            attempts += 1
            r = run_with_fallback(targets, prompt + feedback,
                                  max_tokens=job.get("max_tokens", 4096),
                                  timeout=job.get("timeout", 600),
                                  system=system or None, race=race)
            rec.update({"model": r.get("model"), "secs": r.get("secs"),
                        "tried": r.get("tried")})
            if not r.get("ok"):
                rec["error"] = r.get("error")
                break
            text = r.get("text") or ""
            rec["chars"] = len(text)
            if not text.strip():
                rec["error"] = "empty body"
                break
            out = _resolve_in(worktree, job.get("out"))
            if out:
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".",
                                exist_ok=True)
                    with open(out, "w", encoding="utf-8") as fh:
                        fh.write(text)
                    rec["out"] = out
                except OSError as e:
                    rec["error"] = f"write {out}: {e}"
                    break
            else:
                rec["text"] = text
            ver = _verify(job, worktree, job.get("_repo") or repo or "")
            if not ver.get("ok"):
                rec["error"] = "verify failed: %s" % ver.get("cmd")
                feedback = (
                    "\n\n--- VERIFY FAILED (attempt %d/%d) ---\n"
                    "command: %s\nexit: %s\noutput:\n%s\n\n"
                    "Rewrite the file so this command passes."
                    % (attempts, total, ver.get("cmd"), ver.get("exit_code"),
                       (ver.get("output") or "")[-2000:]))
                continue
            if not want_review:
                rec["ok"] = True
                rec["error"] = None
                rec["_text"] = text
                break
            rounds = max(0, rounds - 1)
            rev = _review(job, worktree, rec.get("out"), text, base, my_pool)
            rec["reviews"].append(rev)
            if rev.get("ok"):
                rec["ok"] = True
                rec["error"] = None
                rec["_text"] = text
                break
            blockers = rev.get("blocking") or []
            rec["error"] = "review rejected: %s" % ("; ".join(blockers) or "no detail")
            feedback = ("\n\n--- REJECTED BY REVIEWER (attempt %d/%d) ---\n"
                        "Your artifact was rejected. Blocking defects:\n%s\n\n%s\n\n"
                        "Rewrite the file so none of these remain."
                        % (attempts, total,
                           "\n".join("- " + b for b in blockers),
                           rev.get("notes") or ""))
        rec["verify"] = ver
        rec["review"] = rev
        rec["attempts"] = attempts
        if worktree:
            rec["worktree"] = worktree
            rec["branch"] = job.get("_branch")
            commit = git_commit_worktree(worktree, f"mr[{run_id}] {jid}")
            rec["commit"] = commit
            if commit.get("ok"):
                rec["files_changed"] = commit.get("files", [])
            elif rec["ok"]:
                rec["ok"] = False
                rec["error"] = commit.get("error")
        return rec

    t0 = time.time()
    results = []
    pending = {job["id"]: (job, wt) for job, wt, _i in prepared}
    while pending:
        ready = []
        for jid, (job, _wt) in pending.items():
            deps = [str(d) for d in (job.get("depends_on") or [])]
            if all(d in ctx for d in deps):
                ready.append(jid)
        if not ready:
            for jid in list(pending):
                results.append({"id": jid, "ok": False,
                                "error": "unresolvable depends_on (cycle)"})
                pending.pop(jid)
            break
        wave = []
        for jid in ready:
            job, wt = pending.pop(jid)
            _seed_worktree(job, wt)
            wave.append((job, wt))
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
            wres = list(ex.map(run, wave))
        for rec in wres:
            text = rec.pop("_text", "")
            job_repo = next((p[0].get("_repo") or "" for p in prepared
                             if p[0]["id"] == rec["id"]), "")
            src_job = next((p[0] for p in prepared if p[0]["id"] == rec["id"]), {})
            ctx[rec["id"]] = {"out": rec.get("out"), "text": text,
                              "repo": job_repo,
                              "rel": str(src_job.get("out") or "")}
            results.append(rec)

    order = {jid: i for i, jid in enumerate(ids)}
    results.sort(key=lambda r: order.get(r["id"], len(order)))

    overlaps = []
    touched = {}
    for r in results:
        for f in r.get("files_changed") or []:
            touched.setdefault(f, []).append(r["id"])
    for f, jids in touched.items():
        if len(jids) > 1:
            overlaps.append({"file": f, "jobs": jids})
    if isolation["enabled"]:
        isolation["merge_hint"] = ("pick winners, then call agent_merge("
                                   "repo=PATH, branches='a,b')")
    failed = [r["id"] for r in results if not r["ok"]]
    return json.dumps({"ok": not failed, "jobs": len(results),
                       "failed": failed, "secs": round(time.time() - t0, 1),
                       "health": report, "isolation": isolation,
                       "overlaps": overlaps, "out_conflicts": out_conflicts,
                       "results": results},
                      indent=1, ensure_ascii=False)

@mcp.tool()
def plan(jobs_json: str, parallel: int = 4, preflight: bool = True,
         repo: str = "", isolate: bool = False, brief: str = "",
         race: int = 2, review: bool = True, reviewer: str = "",
         review_rounds: int = 1) -> str:
    """Run a fanout plan. Each job writes its output to `out`.

    Jobs may declare depends_on (run in waves, ancestors' artifacts are passed
    down), verify (shell command that must exit 0, failures retried), and share
    a `brief` prepended to every system prompt. After a job finishes, a
    separate reviewer model reads the artifact; a rejected job is re-run with
    the reviewer's blocking defects in its prompt. See _plan_impl.
    """
    return _plan_impl(jobs_json, parallel=parallel, preflight=preflight,
                      repo=repo, isolate=isolate, brief=brief, race=race,
                      review=review, reviewer=reviewer,
                      review_rounds=review_rounds)


REVIEWER_PROMPT = """You review ONE artifact written by another agent.

You do not rewrite it and you do not praise it. Read the task, read the
artifact, decide whether the task is actually done.

Return ONLY a JSON object, no prose, no fences:
{"ok": true|false,
 "blocking": ["one line per concrete defect that must be fixed"],
 "notes": "one short paragraph, only when ok is false"}

Set ok=false ONLY for defects that make the artifact wrong or unusable:
syntax or type errors, imports that do not resolve, a function the task
required but is missing, wrong output path or file name, prose where code was
required, a test that does not test what it claims, an API the task specified
implemented with a different name or signature.

Do not set ok=false for style, naming taste, missing comments, missing
optional features, or "could be nicer". If it works and matches the task, ok
is true. An empty or unreadable artifact is blocking.

--- TASK GIVEN TO THE AGENT ---
__TASK__

--- ARTIFACT (__PATH__) ---
__ARTIFACT__"""


AGENT_DISCIPLINE = (
    "You are one agent in a parallel build. Do the task and nothing else. "
    "Write the real artifact to the output path: source code for .go/.py/.js, "
    "prose only for .md. Never reply with narration, progress notes, or "
    "explanations of what you are about to do -- the file content IS your "
    "answer. No markdown fences around source files."
)


PLANNER_PROMPT = """You decompose a goal into parallel agent jobs.

Return ONLY a JSON object, no prose, no markdown fences:
{"agents":[{"id":"short-slug","role":"code|review|reason|design|research|fast",
            "task":"precise self-contained instruction for that agent",
            "out":"relative/path/to/artifact.ext",
            "depends_on":["id-of-an-earlier-agent"],
            "verify":"shell command that must exit 0 in that agent's workspace",
            "review":true}]}

Rules:
- Between __MIN__ and __MAX__ agents. Use the FEWEST that does the job well:
  every extra agent adds a merge conflict. One agent is a valid answer.
- One agent per independent workstream that can run in parallel.
- `task` must stand alone: the agent sees only that text, never the goal.
- `out` is the file that agent writes, relative to its workspace. Give
  different agents different paths. If two must edit one file, say so in the
  task and expect a reported overlap.
- `role` picks the model class: code writes programs, review audits existing
  work, reason analyses/plans, design handles UI and copy, research finds
  facts, fast handles trivial work.
- `depends_on` is the only thing that creates ordering. List the ids whose
  artifacts this agent consumes; it starts after they finish and its prompt
  contains their files. Do not describe a dependency in prose -- declare it.
- `verify` is optional but valuable: a shell command run in that agent's
  workspace after it writes, for example "go build ./..." or
  "python3 -m pytest -q". On failure the agent sees the output and retries.
  Add one whenever the artifact is mechanically checkable.
- `review` defaults to true: a different model reads the artifact and rejects
  it when the task is not actually done, sending the defects back to the agent.
  Set it false only for artifacts nothing can meaningfully check (raw research
  notes).

GOAL:
__GOAL__"""


def _extract_json(text):
    """Pull the first JSON object out of a model reply."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\n```\s*$", "", t).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return None


@mcp.tool()
def spawn_agents(goal: str, max_agents: int = 5, min_agents: int = 1,
                 repo: str = "", isolate: bool = True, parallel: int = 3,
                 planner: str = "", max_tokens: int = 4096,
                 timeout: int = 600, system: str = "",
                 dry_run: bool = False, brief: str = "",
                 review: bool = True, reviewer: str = "",
                 review_rounds: int = 1) -> str:
    """Let a planner model decide how many agents to run and what each one does.

    goal: what must be accomplished, in plain language.
    max_agents: hard cap on fan-out (clamped 1..12). The planner may use fewer.
    min_agents: floor it should not go below (clamped 1..max_agents).
    planner: model or role that writes the split (default: the `reason` role).
    repo / isolate / parallel: same meaning as in plan(); each agent gets its
        own git worktree and branch when isolate is on.
    dry_run: return the decomposed jobs without spending an agent call.
    review / reviewer / review_rounds: after each agent writes, a separate
        reviewer model checks its artifact; a rejected agent redoes the work
        with the reviewer's blocking defects attached. review=false turns the
        check off for the whole run.
    """
    if not goal or not goal.strip():
        return json.dumps({"ok": False, "error": "goal is required"})
    max_agents = max(1, min(int(max_agents or 1), 12))
    min_agents = max(1, min(int(min_agents or 1), max_agents))

    planner_models = resolve(planner or "reason")
    planner_models, health = filter_alive(planner_models)
    if not planner_models:
        return json.dumps({"ok": False, "stage": "plan",
                           "error": "no planner model available"})

    prompt = (PLANNER_PROMPT.replace("__MIN__", str(min_agents))
              .replace("__MAX__", str(max_agents))
              .replace("__GOAL__", goal.strip()))
    cls = run_with_fallback(planner_models, prompt, max_tokens=2000,
                            timeout=min(timeout, 180), temperature=0,
                            system=system or None)
    if not cls.get("ok"):
        return json.dumps({"ok": False, "stage": "plan",
                           "error": cls.get("error"), "tried": cls.get("tried"),
                           "health": health}, indent=1, ensure_ascii=False)

    parsed = _extract_json(cls.get("text"))
    if not parsed or not isinstance(parsed.get("agents"), list):
        return json.dumps({"ok": False, "stage": "parse",
                           "error": "planner did not return valid JSON",
                           "raw": (cls.get("text") or "")[:2000],
                           "planner": cls.get("model")},
                          indent=1, ensure_ascii=False)

    seen, jobs = set(), []
    for i, a in enumerate(parsed["agents"][:max_agents]):
        if not isinstance(a, dict):
            continue
        jid = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(a.get("id") or f"agent-{i + 1}"))
        if jid in seen:
            jid = f"{jid}-{i + 1}"
        seen.add(jid)
        role = a.get("role") if a.get("role") in ROLES else "code"
        task = str(a.get("task") or "").strip()
        if not task:
            continue
        job = {"id": jid, "role": role, "prompt": task,
               "out": str(a.get("out") or f"{jid}.md"),
               "max_tokens": max_tokens, "timeout": timeout,
               "isolate": bool(isolate),
               "depends_on": [str(x) for x in (a.get("depends_on") or [])],
               "system": system or AGENT_DISCIPLINE,
               "review": bool(a.get("review", True)) and bool(review),
               "review_rounds": review_rounds}
        if reviewer:
            job["reviewer"] = reviewer
        if a.get("verify"):
            job["verify"] = str(a["verify"])
        if repo:
            job["repo"] = repo
        jobs.append(job)
    if not jobs:
        return json.dumps({"ok": False, "stage": "parse",
                           "error": "planner returned no usable agents",
                           "raw": (cls.get("text") or "")[:2000]},
                          indent=1, ensure_ascii=False)

    # Two agents writing one path silently overwrite each other. Make the
    # planner's paths unique before spending a single model call.
    seen_out = set()
    for job in jobs:
        o = job.get("out") or ""
        if o in seen_out:
            root, ext = os.path.splitext(o)
            n = 2
            while "%s-%d%s" % (root, n, ext) in seen_out:
                n += 1
            job["out"] = "%s-%d%s" % (root, n, ext)
        seen_out.add(job["out"])

    summary = [{"id": j["id"], "role": j["role"], "out": j["out"],
                "task": j["prompt"][:400]} for j in jobs]
    if dry_run:
        return json.dumps({"ok": True, "dry_run": True, "agents": len(jobs),
                           "requested": {"min": min_agents, "max": max_agents},
                           "planner": cls.get("model"), "plan_health": health,
                           "plan": summary}, indent=1, ensure_ascii=False)

    res = json.loads(_plan_impl(json.dumps(jobs), parallel=parallel,
                                preflight=True, repo=repo,
                                isolate=bool(isolate), brief=brief,
                                review=review, reviewer=reviewer,
                                review_rounds=review_rounds))
    return json.dumps({"ok": res.get("ok", False), "agents": len(jobs),
                       "requested": {"min": min_agents, "max": max_agents},
                       "planner": cls.get("model"), "plan_health": health,
                       "plan": summary, "failed": res.get("failed"),
                       "secs": res.get("secs"), "isolation": res.get("isolation"),
                       "overlaps": res.get("overlaps"),
                       "results": res.get("results")},
                      indent=1, ensure_ascii=False)


@mcp.tool()
def health_check(models: str = "", parallel: int = 8, timeout: int = 60,
                 force: bool = False) -> str:
    """Pre-flight: which models answer right now. Reported to the agent before any run.

    models: comma list or role name; empty = every configured model.
    force: ignore the cache and re-ping (cache TTL ~10 min).
    """
    if models:
        targets = resolve(models)
    else:
        targets = sorted({m for i in config_models().values() for m in i["models"]})
    if not targets:
        return json.dumps({"ok": False, "error": "no models"}, ensure_ascii=False)
    h, from_cache = health_snapshot(targets, parallel=parallel, timeout=timeout,
                                    force=force)
    dead = {d["model"] for d in h["dead"]}
    mine = [m for m in targets if m in set(h["alive"]) or m in dead]
    alive = [m for m in mine if m not in dead]
    dead_here = [d for d in h["dead"] if d["model"] in set(targets)]
    return json.dumps({"alive": alive, "dead": dead_here,
                       "counts": {"alive": len(alive), "dead": len(dead_here),
                                  "total": len(mine)},
                       "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                                                   time.localtime(h["ts"])),
                       "cached": from_cache, "ttl_secs": HEALTH_TTL},
                      indent=1, ensure_ascii=False)


@mcp.tool()
def probe(models: str = "", parallel: int = 8, timeout: int = 60) -> str:
    """Health-check models with a 1-token ping. models: comma list; empty = all configured."""
    if models:
        targets = resolve(models)
    else:
        targets = sorted({m for i in config_models().values() for m in i["models"]})
    if not targets:
        return json.dumps({"ok": False, "error": "no models"})

    alive, dead, unknown = probe_models(targets, parallel=parallel, timeout=timeout)
    health = os.path.join(HOME, "agent-fanout", "health.json")
    os.makedirs(os.path.dirname(health), exist_ok=True)
    json.dump({"alive": alive, "dead": dead,
               "unknown": [u["model"] for u in unknown]},
              open(health, "w"), indent=1)
    return json.dumps({"alive": alive, "dead": dead,
                       "unknown": [u["model"] for u in unknown],
                       "health_file": health},
                      indent=1, ensure_ascii=False)



# --------------------------------------------------------------- git isolation
SCRATCH = os.path.join(HOME, "cache", "scratch")


def git_available():
    try:
        r = subprocess.run(["git", "--version"], capture_output=True, text=True,
                           timeout=10)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _git(args, cwd, timeout=180):
    r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True,
                       timeout=timeout)
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


def _commit_args(message):
    return ["-c", "user.name=modelrouter", "-c", "user.email=modelrouter@local",
            "commit", "-q", "-m", message]


def git_ensure_repo(path, base_branch="main"):
    """Make path a git repo with at least one commit (worktrees need a HEAD)."""
    path = os.path.abspath(os.path.expanduser(path))
    if not git_available():
        return {"ok": False, "error": "git is not installed"}
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        return {"ok": False, "error": f"cannot create {path}: {e}"}
    rc, _, _ = _git(["rev-parse", "--is-inside-work-tree"], path)
    created = False
    if rc == 0:
        # `path` may be a subdirectory of a repo; worktrees and branches belong
        # to the root, and reporting the subdirectory mislabels the repo.
        rc_top, top, _ = _git(["rev-parse", "--show-toplevel"], path)
        if rc_top == 0 and top.strip():
            path = top.strip()
    if rc != 0:
        rc, _, err = _git(["init", "-q", "-b", base_branch], path)
        if rc != 0:
            rc, _, err = _git(["init", "-q"], path)
            if rc != 0:
                return {"ok": False, "error": f"git init failed: {err}"}
        created = True
    rc, _, _ = _git(["rev-parse", "--verify", "--quiet", "HEAD"], path)
    committed = False
    if rc != 0:
        _git(["add", "-A"], path)
        rc, _, err = _git(_commit_args("modelrouter: initial commit"), path)
        if rc != 0:
            # Nothing to stage in an empty dir: commit an empty root instead,
            # worktrees only need a HEAD to exist.
            rc, _, err = _git(_commit_args("modelrouter: initial commit")
                              + ["--allow-empty"], path)
            if rc != 0:
                return {"ok": False,
                        "error": f"initial commit failed: {err or '(no stderr)'}"}
        committed = True
    rc, head, _ = _git(["rev-parse", "HEAD"], path)
    rc2, branch, _ = _git(["rev-parse", "--abbrev-ref", "HEAD"], path)
    return {"ok": True, "repo": path, "created": created,
            "initial_commit": committed, "head": head,
            "branch": branch if rc2 == 0 else "HEAD"}


def git_worktree_add(repo, path, branch, base_ref):
    """Private checkout for one agent, so parallel jobs never share a tree."""
    if os.path.exists(path):
        if os.listdir(path):
            _git(["worktree", "remove", "--force", path], repo)
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rc, _, err = _git(["worktree", "add", "-q", "-b", branch, path, base_ref], repo)
    if rc != 0:
        rc, _, err = _git(["worktree", "add", "-q", path, base_ref], repo)
        if rc != 0:
            return {"ok": False, "error": err or "worktree add failed"}
        # `-b` failed (the branch exists). Reuse it instead of leaving a
        # detached HEAD, which agent_merge cannot resolve as a branch.
        rc_b, _, _ = _git(["checkout", "-q", "-B", branch], path)
        rc, br, _ = _git(["rev-parse", "--abbrev-ref", "HEAD"], path)
        branch = (br or "").strip() or branch
        if not rc_b == 0 and branch == "HEAD":
            return {"ok": False,
                    "error": "worktree %s has a detached HEAD" % path}
    return {"ok": True, "worktree": path, "branch": branch}


def git_commit_worktree(path, message):
    rc, status, err = _git(["status", "--porcelain"], path)
    if rc != 0:
        return {"ok": False, "error": err}
    if not status:
        return {"ok": True, "committed": False, "files": []}
    _git(["add", "-A"], path)
    rc, _, err = _git(_commit_args(message), path)
    if rc != 0:
        return {"ok": False, "error": f"commit failed: {err}"}
    rc, out, _ = _git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"], path)
    return {"ok": True, "committed": True,
            "files": [f for f in out.splitlines() if f.strip()]}


def git_changed_files(repo, base_ref, branch):
    rc, out, _ = _git(["diff", "--name-only", base_ref, branch], repo)
    return [f for f in out.splitlines() if f.strip()] if rc == 0 else []


def git_merge_branch(repo, branch, base_branch="", message=""):
    """Merge one agent branch. On conflict: abort, report the files, never force."""
    if base_branch:
        rc, _, _ = _git(["rev-parse", "--verify", base_branch], repo)
        if rc != 0:
            return {"ok": False, "branch": branch,
                    "error": f"unknown base branch {base_branch}"}
        cur = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)[1]
        if cur != base_branch:
            rc, _, err = _git(["checkout", "-q", base_branch], repo)
            if rc != 0:
                return {"ok": False, "branch": branch,
                        "error": f"checkout {base_branch} failed: {err}"}
    rc, _, err = _git(["-c", "user.name=modelrouter",
                       "-c", "user.email=modelrouter@local",
                       "merge", "--no-ff", "-m",
                       message or f"merge {branch}", branch], repo)
    if rc == 0:
        return {"ok": True, "branch": branch, "merged": True}
    rc2, conflicts, _ = _git(["diff", "--name-only", "--diff-filter=U"], repo)
    _git(["merge", "--abort"], repo)
    first = (err.splitlines() or ["merge failed"])[0]
    return {"ok": False, "branch": branch, "merged": False,
            "conflicts": [c for c in conflicts.splitlines() if c.strip()],
            "error": first}


def git_worktrees(repo):
    rc, out, _ = _git(["worktree", "list", "--porcelain"], repo)
    if rc != 0:
        return []
    trees, cur = [], {}
    for line in out.splitlines():
        if line.startswith("worktree "):
            if cur:
                trees.append(cur)
            cur = {"path": line[9:]}
        elif line.startswith("branch "):
            cur["branch"] = line[7:].replace("refs/heads/", "")
        elif line == "detached":
            cur["branch"] = "(detached)"
    if cur:
        trees.append(cur)
    return trees


@mcp.tool()
def git_status(repo: str = "") -> str:
    """Report git availability and a repo's state: branch, head, dirty files, worktrees.

    repo: path to inspect; empty = current working directory.
    """
    if not git_available():
        return json.dumps({"git": False, "error": "git is not installed"},
                          ensure_ascii=False)
    path = os.path.abspath(os.path.expanduser(repo)) if repo else os.getcwd()
    rc, root, err = _git(["rev-parse", "--show-toplevel"], path)
    if rc != 0:
        return json.dumps({"git": True, "repo": path, "is_repo": False,
                           "error": err}, indent=1, ensure_ascii=False)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)[1]
    head = _git(["rev-parse", "--short", "HEAD"], root)[1]
    dirty = _git(["status", "--porcelain"], root)[1]
    return json.dumps({"git": True, "is_repo": True, "repo": root,
                       "branch": branch, "head": head,
                       "dirty": [x for x in dirty.splitlines() if x.strip()],
                       "worktrees": git_worktrees(root)},
                      indent=1, ensure_ascii=False)


@mcp.tool()
def git_init(path: str, base_branch: str = "main") -> str:
    """Make `path` a git repo with at least one commit, so agents can use worktrees."""
    return json.dumps(git_ensure_repo(path, base_branch), indent=1,
                      ensure_ascii=False)


@mcp.tool()
def agent_merge(repo: str, branches: str, base_branch: str = "",
                cleanup: bool = False) -> str:
    """Merge agent branches the main model explicitly chose, one at a time.

    branches: comma-separated branch names (from a plan's isolation report).
    base_branch: branch to merge into; empty = whatever the repo has checked out.
    cleanup: remove the worktree of each branch that merged cleanly.

    Conflicting merges are aborted and reported with the conflicting files —
    nothing is force-resolved here. The main model picks the winner and retries.
    """
    repo = os.path.abspath(os.path.expanduser(repo))
    rc, root, err = _git(["rev-parse", "--show-toplevel"], repo)
    if rc != 0:
        return json.dumps({"ok": False, "error": f"not a git repo: {err}"},
                          ensure_ascii=False)
    names = [b.strip() for b in branches.split(",") if b.strip()]
    if not names:
        return json.dumps({"ok": False, "error": "no branches given"},
                          ensure_ascii=False)
    trees = {t.get("branch"): t.get("path") for t in git_worktrees(root)}
    results, merged = [], []
    for b in names:
        rc, _, err = _git(["rev-parse", "--verify", b], root)
        if rc != 0:
            results.append({"branch": b, "ok": False, "error": "unknown branch"})
            continue
        r = git_merge_branch(root, b, base_branch)
        results.append(r)
        if r.get("ok"):
            merged.append(b)
            if cleanup and trees.get(b):
                _git(["worktree", "remove", "--force", trees[b]], root)
    return json.dumps({"ok": len(merged) == len(names), "repo": root,
                       "merged": merged,
                       "failed": [r for r in results if not r.get("ok")],
                       "results": results}, indent=1, ensure_ascii=False)


def _boot_probe():
    """Probe the whole pool a moment after start, off the MCP handshake path."""
    time.sleep(float(os.environ.get("MODELROUTER_BOOT_DELAY", "2")))
    try:
        r = pool_probe(reason="startup")
        print("pool boot: " + json.dumps(r, ensure_ascii=False), file=sys.stderr,
              flush=True)
    except Exception as e:  # never let the boot probe kill the server
        print(f"pool boot failed: {type(e).__name__}: {e}", file=sys.stderr,
              flush=True)


if __name__ == "__main__":
    if os.environ.get("MODELROUTER_BOOT_PROBE", "1") not in ("0", "false", "no"):
        threading.Thread(target=_boot_probe, name="pool-boot",
                         daemon=True).start()
    mcp.run()