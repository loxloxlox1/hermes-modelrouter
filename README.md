# hermes-modelrouter

A local MCP server that turns every model Hermes Agent knows about into a
role-routed pool with health checking, git worktree isolation, and a
merge gate that refuses to auto-resolve conflicts.

It reads `~/.hermes/config.yaml` for the model inventory and
`~/.hermes/agent-fanout/routing.yaml` for role assignments, so there is nothing
to configure twice: whatever Hermes can reach, this server can route.

## Why

An orchestrating model should not spend its own context generating every file.
It should decompose the work, hand pieces to cheaper or stronger models, and
check the result. That needs three things this server provides:

1. **Role routing** — `code`, `review`, `reason`, `design`, `research`, `fast`
   each map to an ordered fallback chain. First model that answers wins.
2. **Health gating** — models get pinged, dead ones leave the pool, so a broken
   endpoint costs one startup probe instead of a request timeout on every job.
3. **Isolation** — parallel jobs each get their own git worktree and branch. The
   server reports which files two agents both touched, and the orchestrator
   picks the winner. Nothing merges automatically.

## Install

Requires Python 3.10+, `uv`, and `git` (git is optional — isolation degrades to
shared-directory writes and says so in its report).

```bash
cp -r hermes-modelrouter ~/.hermes/mcp/
```

Register it with Hermes (the config file is write-protected against hand edits,
so use the CLI):

```bash
hermes config set mcp_servers.modelrouter.command uv
hermes config set mcp_servers.modelrouter.args \
  '["run","--with","mcp","python3","/home/<user>/.hermes/mcp/hermes-modelrouter/server.py"]'
hermes config set mcp_servers.modelrouter.timeout 600
```

Restart Hermes or run `/reload-mcp`. Tools appear as `mcp_modelrouter_*`.

### Credentials

Read from `~/.hermes/.env` on first use. Resolution order:

1. `AGENT_FANOUT_KEY`
2. `HERMES_CUSTOM_OSNV_API_KEY`
3. `CUSTOM_PROVIDER_OSNV_KEY`
4. `HERMES_CUSTOM_<host>_<port>_API_KEY` derived from each provider's `base_url`

Point it at any OpenAI-compatible endpoint with `AGENT_FANOUT_BASE`
(default `https://ai-router.l.cd/v1`).

## Tools

### Routing

| Tool | Purpose |
|---|---|
| `list_models(grep, provider)` | every configured model, grouped by provider |
| `list_roles()` | role -> ordered candidates, flagged `in_config` |
| `set_role(role, models)` | rewrite a role's chain (`"a,b,c"`, first alive wins) |
| `auto_route(task, models, preflight)` | classify the task into a role with a cheap model, then run it there |

### Running

| Tool | Purpose |
|---|---|
| `ask(prompt, models, preflight)` | one prompt against a role, a model, or a fallback chain |
| `plan(jobs_json, parallel, preflight, repo, isolate)` | parallel fan-out; each job writes its artifact to disk |
| `spawn_agents(goal, max_agents, min_agents, repo, isolate, dry_run)` | a planner model decides the agent count and each agent's task, then runs them |
| `health_check(models, force)` | cached pre-flight: alive / dead right now |
| `probe(models)` | uncached ping sweep |

### Model pool

| Tool | Purpose |
|---|---|
| `pool_status()` | what the startup probe kept and dropped, per role |
| `pool_refresh(models)` | re-probe on demand |

### Git isolation

| Tool | Purpose |
|---|---|
| `git_status(repo)` | branch, head, dirty files, live worktrees |
| `git_init(path)` | make a path a repo with one commit (worktrees need a HEAD) |
| `agent_merge(repo, branches, base_branch, cleanup)` | merge only the branches you chose; conflicts come back aborted |

## The fan-out loop

```
plan(repo=PATH, isolate=true, jobs_json=[...])
  -> results[].files_changed      what each agent actually wrote
  -> overlaps[]                   files touched by more than one job
decide per overlap
agent_merge(repo=PATH, branches="winner-a,winner-b", cleanup=true)
  -> merged[] / failed[].conflicts
```

`overlaps` exists so the orchestrator decides. Merging every branch blindly is
how you get a file that no model intended to write.

## Letting the planner pick the split

`plan()` takes jobs you already wrote. `spawn_agents()` takes a goal and asks a
planner model to do that part:

```
spawn_agents(goal="...", max_agents=5, min_agents=2, repo=PATH, isolate=true)
  -> agents N                       how many the planner chose
  -> plan[{id, role, out, task}]    the decomposition
  -> results[] / overlaps[]         same shape as plan()
```

The planner sees only the goal and the bounds; it returns JSON. Its rules are in
`PLANNER_PROMPT` and are load-bearing: use the fewest agents that does the job
(every extra agent is a future merge conflict), one agent per independent
workstream, each `task` self-contained, distinct `out` paths.

`dry_run=True` returns the decomposition without spending an agent call — check
the split before paying for it.

Bounds are clamped: `max_agents` 1..12, `min_agents` 1..max_agents. A malformed
planner reply comes back as `stage: "parse"` with the raw text, not a crash.

## Health model

On start the server probes every configured model once and stores the verdict in
`~/.hermes/agent-fanout/model-pool.json`. Dead models are pruned from role
lookups; a model named explicitly is passed through unchanged.

Failures are classified, because they are not equivalent:

- `dead` — HTTP 4xx other than 429, or HTTP 200 with an empty body. Loses its slot.
- `unknown` — 429, 5xx, timeout, connection error. **Keeps** its previous slot,
  since throttling is not death.
- If *every* model fails, the endpoint is down, not the models — the pool is left
  untouched rather than emptied.

Tuning:

| Env var | Default | Meaning |
|---|---|---|
| `MODELROUTER_BOOT_PROBE` | `1` | set `0` to skip the startup probe |
| `MODELROUTER_BOOT_DELAY` | `2` | seconds before the probe fires |
| `MODELROUTER_POOL_TIMEOUT` | `45` | per-model ping timeout |
| `MODELROUTER_PROBE_GAP` | `1.5` | min gap between pings to one endpoint |
| `MODELROUTER_HEALTH_TTL` | `600` | `health_check` cache TTL |

Probe parallelism is deliberately low (3). A parallel burst of pings to one
aggregator returns 429 for most of them, which reads as mass model death.

## Job format

```json
{
  "id": "snake-code",
  "role": "code",
  "out": "snake.py",
  "prompt": "...",
  "system": "Output ONLY code.",
  "max_tokens": 6000,
  "timeout": 300,
  "files": ["src/app.py"],
  "repo": "/path/to/repo",
  "isolate": true
}
```

`out`, `files`, and `prompt_file` resolve inside the job's worktree when
isolation is on. A job that wrote zero bytes counts as failed — an HTTP 200 with
an empty body is not success.

## Tests

The three client scripts are real MCP SDK clients, not hand-rolled JSON-RPC. The
SDK does era detection on the first frames; piping JSON into stdin produces
confusing false failures.

```bash
uv run --with mcp python3 client_test.py          # tools + ask + auto_route
uv run --with mcp python3 client_test_health.py   # pool + preflight
uv run --with mcp python3 client_test_git.py      # worktree isolation + merge
```

`client_test_git.py` creates a throwaway repo under
`~/.hermes/cache/scratch/git-iso-test`, has two jobs deliberately collide on one
file, and asserts that the conflict is reported and aborted rather than resolved.

## Notes and known limits

- Config parsing is hand-rolled (no PyYAML) and reads `providers.*.models` only.
  Models registered outside a `providers:` block are invisible.
- Local providers often declare no `key_env`; the key name is derived from the
  URL as `HERMES_CUSTOM_<host>_<port>_API_KEY`, matching Hermes' convention.
- Worktrees live under `~/.hermes/cache/scratch/modelrouter-wt/<run-id>/` and are
  removed by `agent_merge(cleanup=true)` for branches that merged cleanly.
  Unmerged ones stay until pruned.
- `plan` writes artifacts to disk; text in the reply is a preview, the file on
  disk is the contract.

## License

MIT