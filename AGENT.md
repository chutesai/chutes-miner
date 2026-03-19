# Agent Constraints

**Read this file before making any changes.** These constraints apply to all AI-assisted work in this repository.

## Project Identity

Chutes miner is the software stack for mining on chutes.ai — a permissionless, serverless, AI-centric compute platform. This monorepo contains all components for provisioning GPU nodes, managing chutes (apps), validating hardware, and coordinating with validators. Mining optimizes for cold start times and total compute time; incentives are based on 7-day compute sums.

## Stack (Non-Negotiable)

- **Language**: Python 3.12+
- **Package manager**: Poetry 2.x
- **HTTP services**: FastAPI + Uvicorn
- **Database**: SQLAlchemy + asyncpg (PostgreSQL)
- **Caching / pubsub**: Redis
- **Orchestration**: Kubernetes (k3s recommended)
- **Deployment**: Helm charts, Ansible for provisioning
- **Linting**: Ruff
- **Testing**: pytest, pytest-asyncio
- **CLI**: Typer or Click

Do not introduce alternate frameworks (e.g., Prisma, NextAuth, Firebase). Stay within this stack.

## Hard Rules

- **Never install a new dependency** without discussion first
- **Never modify database schemas** without showing the migration plan (dbmate migrations live in `src/<package>/.../migrations/`)
- **All packages live under `src/`**, Docker configs under `docker/`, tests under `tests/`
- **Environment variables** go in `.env`, `local.env`, or Helm values — never hardcoded
- **90% test coverage target** — if you change code, add tests for it
- **Run `make lint-local` and `make reformat`** before committing
- **Use Make commands for tooling** — never run `python`, `pytest`, or `ruff` directly. The global Python interpreter does not have project dependencies installed. Use `make test-local`, `make lint-local`, `make reformat` instead.
- **No new top-level packages** under `src/` without discussion
- **Version bumps** — each package has a `VERSION` file; update when releasing

## Patterns

- **Module-level imports** — Prefer imports at the top of the file. Do not use imports inside functions or methods without explicit approval; keep imports at module level for clarity.
- **Async-first**: Use `async def`, `asyncio`, async DB sessions. Avoid blocking calls in request handlers
- **Pydantic models** for request/response schemas; use `chutes_common.schemas` for shared types
- **Shared utilities** go in `chutes-common` — auth, Redis, K8s client, monitoring, schemas
- **One concern per module** — keep files focused; split when they grow large
- **Follow existing naming** — check neighboring files and packages for conventions
- **Meaningful method names** — Avoid `run_...` or `do_...` prefixes. Methods clearly execute; use verbs that describe the action (e.g. `verify_server`, `bootstrap_server`, not `run_bootstrap` or `do_verify`).
- **Error handling**: Use custom exceptions from `chutes_common.exceptions` or package-specific `exceptions.py`; avoid bare `except`


## Architecture Overview

| Package | Purpose |
|---------|---------|
| **chutes-miner** | Miner API: server/inventory, websocket to validator, registry auth, Gepetto (chute management). Core mining logic. |
| **chutes-miner-cli** | CLI for miner ops: `add-node`, `sync-kubeconfig`, TEE helpers. |
| **chutes-agent** | Agent on GPU nodes: monitors chutes, exposes kubeconfig/config. Runs on each GPU server. |
| **chutes-monitor** | Monitors chutes in member clusters. |
| **chutes-common** | Shared lib: schemas, auth, Redis, K8s, monitoring. |
| **chutes-registry** | Local registry proxy for miner nodes. |
| **chutes-cache-cleaner** | HuggingFace/CivitAI cache cleanup. |
| **graval-bootstrap** | GraVal GPU validation bootstrap. |

Helm charts: `charts/chutes-miner`, `charts/chutes-miner-gpu`, `charts/chutes-monitoring`.

## Development Commands

**Always use Make commands** — they run tools via the project's Poetry venv. Do not invoke `python`, `pytest`, or `ruff` directly; the global interpreter lacks project dependencies.

```bash
make help              # List all targets
make list-packages     # List packages in monorepo
make venv              # Create shared venv in .venv/ (run once; poetry.toml sets in-project)
make test-local        # Run tests via venv (or: make test-local <package>)
make lint-local        # Run Ruff via venv (or: make lint-local <package>)
make reformat          # Format code
make build             # Build Docker images (or: make build <package>)
make ci                # Full CI pipeline
```

For chutes-miner dev environment:

```bash
cd docker/chutes-miner
docker compose up -d postgres redis
docker compose up api
```

API at `http://localhost:8080`.
