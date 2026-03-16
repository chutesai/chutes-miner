# Spec Templates

Prompt contract templates for designing and implementing features, bugfixes, and refactors. These specs standardize AI agent interaction and double as design documentation.

## How to Use

1. **Copy** the appropriate template from `templates/` (feature, bugfix, or refactor)
2. **Fill in** each section before starting work — the Goal, Constraints, Output Format, and Failure Conditions are the core prompt contract
3. **Use as AI prompt** — paste the filled spec into your agent session; AGENT.md is always the constraints baseline
4. **Save as documentation** — commit completed specs to the repo for design decision history

## Naming Convention

Save completed specs as `YYYY-MM-DD-short-name.md` in `docs/specs/` (or a `completed/` subfolder). Example: `2025-03-16-agent-monitor-retry.md`.

## AGENT.md

All specs implicitly include [AGENT.md](../../AGENT.md) at the repo root as the permanent constraints layer. Task-specific constraints in the spec supplement, not replace, those rules.
