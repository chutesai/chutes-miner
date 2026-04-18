# Feature Spec: `purge-server` CLI Command

**Date**: 2025-07-18  
**Status**: draft

---

## Context

The `chutes-miner-cli` package currently offers `purge-deployments` (purge all
deployments across every server) and `purge-deployment` (purge a single
deployment by ID, or — misleadingly — all deployments on a server via
`--node-id`). There is no clean, single-purpose command to purge all
deployments from a specific server.

The `purge-deployment --node-id` path is semantically broken: the singular
command name implies one deployment, but the flag actually purges **all**
deployments on the targeted server. A dedicated `purge-server` command makes
the scope obvious and aligns with the existing naming convention (`delete-node`,
`lock`, `unlock`).

The backend already supports the operation — `DELETE /servers/{id_or_name}/deployments`
resolves by server name **or** server ID via the `_get_server` helper.

- **Packages affected**: `chutes-miner-cli`
- **Key files**:
  - `src/chutes-miner-cli/chutes_miner_cli/cli.py`
  - `tests/chutes-miner-cli/test_purge_deployments.py`
  - `tests/chutes-miner-cli/fixtures/api_fixtures.py`
  - `tests/chutes-miner-cli/fixtures/cli_fixtures.py`
- **Dependencies**:
  - Backend endpoint `DELETE /servers/{id_or_name}/deployments` in
    `src/chutes-miner/chutes_miner/api/server/router.py` — already exists,
    no changes required.
  - Backend helper `_get_server` resolves via
    `or_(Server.name == id_or_name, Server.server_id == id_or_name)`.

---

## Design Decisions

- **New dedicated command** (`purge-server`) rather than overloading
  `purge-deployments` with an optional server argument. The two operations have
  different scopes (all servers vs one server) and deserve distinct commands.
- **`--name` accepts a server name or server ID** — matches backend behavior
  and is consistent with `delete-node`, `lock`, `unlock` which all pass through
  `_get_server(id_or_name)`. The help text makes this explicit.
- **Reuses `delete_preflight`** before purging — warns the user if active jobs
  exist on the server, consistent with `purge-deployment --node-id` and
  `delete-node`.
- **No backend changes** — the API already supports the operation fully.
- **Existing commands are untouched** — `purge-deployments` and
  `purge-deployment` remain as-is to avoid breaking existing scripts/workflows.

---

## API Changes

- **New endpoints**: None
- **Schema changes**: None
- **Migrations**: None

---

## Goal

Success = A user can run:

```
chutes-miner purge-server --name <server-name-or-id>
```

and all non-job deployments on that specific server are purged. Specifically:

1. `delete_preflight` is called first; if active jobs exist the user is warned
   and must confirm before proceeding.
2. `DELETE /servers/{name}/deployments` is called on the miner API.
3. The JSON response is printed showing which deployments were purged.
4. The command works when given a server **name** or a server **ID**.
5. Unit tests pass covering: happy path by name, happy path by ID,
   preflight-denied path, and CLI registration.

---

## Constraints

- CLI-only change in `chutes-miner-cli`; no backend (miner API) modifications.
- Must include `delete_preflight` check before purging.
- Follow existing CLI patterns: `asyncio.run()` wrapper,
  `sign_request(hotkey, purpose="management")`,
  `aiohttp.ClientSession(raise_for_status=True)`.
- `--name` is a **required** option; omitting it must produce a clear Typer
  error.
- Tests must follow the existing fixture/mock pattern in
  `tests/chutes-miner-cli/`.

---

## Output Format

1. **`src/chutes-miner-cli/chutes_miner_cli/cli.py`** — Add `purge_server`
   function and register as `purge-server` command.
2. **`tests/chutes-miner-cli/test_purge_server.py`** (new file) — Tests:
   happy path by name, happy path by server ID, preflight-denied (no DELETE
   fired), CLI integration via `CliRunner`.
3. **`tests/chutes-miner-cli/fixtures/api_fixtures.py`** — Add
   `mock_purge_server_response` fixture (same response shape as
   `mock_purge_deployments_response`: `{"status": "initiated",
   "deployments_purged": [...]}`).
4. **`src/chutes-miner-cli/VERSION`** — Bump patch version `0.4.0` → `0.4.1`.
5. **`src/chutes-miner-cli/pyproject.toml`** — Bump version to match.

---

## Failure Conditions

- `purge-server` without `--name` does not produce a clear error.
- Active jobs on the server are silently purged without a preflight warning.
- Existing commands (`purge-deployments`, `purge-deployment`) behavior is
  modified in any way.
- Passing a server ID (not a name) fails or hits a different endpoint.
- Preflight denial does not abort the purge (DELETE still fires).
- Tests don't pass or coverage drops.

---

## Rollout Notes

- CLI-only change, no infrastructure / Ansible / feature-flag changes.
- Fully backward compatible: no existing commands are modified.
- Version bump: `0.4.0` → `0.4.1` in `VERSION` and `pyproject.toml`.
- Consider deprecating `purge-deployment --node-id` in a future release with a
  stderr warning pointing users to `purge-server`.

---

## Implementation Plan

| Step | Action | Files |
|------|--------|-------|
| 1 | Add `purge_server` function in `cli.py` with required `--name` option (help: "Name or ID of the server to purge deployments from"), standard `--hotkey` and `--miner-api` options. Calls `delete_preflight`, then `DELETE /servers/{name}/deployments`. Prints JSON response. | `src/chutes-miner-cli/chutes_miner_cli/cli.py` |
| 2 | Register the command near the existing purge commands: `app.command(name="purge-server", help="Purge all deployments from a specific server")(purge_server)` | `src/chutes-miner-cli/chutes_miner_cli/cli.py` |
| 3 | Add `mock_purge_server_response` fixture | `tests/chutes-miner-cli/fixtures/api_fixtures.py` |
| 4 | Create test file with four tests: (a) happy path by name, (b) happy path by server ID, (c) preflight returns False → no DELETE, (d) CLI integration via CliRunner | `tests/chutes-miner-cli/test_purge_server.py` |
| 5 | Bump version `0.4.0` → `0.4.1` | `src/chutes-miner-cli/VERSION`, `src/chutes-miner-cli/pyproject.toml` |
| 6 | Run `make lint-local chutes-miner-cli` and `make test-local chutes-miner-cli` | — |

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Confusion between `purge-server` and `purge-deployment --node-id` | Medium | They hit the same backend endpoint. `purge-server` is the clean path forward. Deprecate `--node-id` later. |
| User passes a name that matches a different server's ID | Very low | Same risk exists in `lock`, `unlock`, `delete-node`. Backend resolves via `or_()` — name-first is acceptable. |
| `delete_preflight` network error crashes CLI | Low | Same behavior as `delete-node` and `purge-deployment --node-id`. Acceptable for CLI. |
