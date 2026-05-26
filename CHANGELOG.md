# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.5.0] — chutes-miner · 2026-05-26

### Added
- **Static registry hostname migration support** (`STATIC_REGISTRY_MIN_VERSION` env var).
  TEE VMs are migrating from a per-validator registry hostname
  (`<validator>.localregistry.chutes.ai`) to a single static hostname
  (`localregistry.chutes.ai`). While both VM generations coexist the miner must
  route image pulls to the correct hostname based on the VM's version.
  - `Gepetto` now maintains an in-memory `remote_server_versions` dict (keyed by
    `server_id`) populated each reconcile cycle from `GET /miner/servers/` on the
    validator. The validator is the authoritative source of VM version via TDX
    quote reconciliation; the miner never persists this.
  - `build_chute_job` selects `localregistry.chutes.ai` when
    `semver(vm_version) >= STATIC_REGISTRY_MIN_VERSION`, falling back to the
    legacy validator-prefixed hostname for older VMs.
  - The reconciler logs a warning listing servers still on the legacy hostname and
    an info line when all servers have migrated — giving operators a clear signal
    for when transition code can be removed.
  - All transition code is tagged `# TRANSITION CODE` with inline removal
    instructions.

### Changed
- Ansible (`k3s` and `microk8s` roles) now configures `localregistry.chutes.ai`
  as the static `registry_hostname` instead of the per-validator subdomain.
- `README.md` updated to reflect the new static registry hostname.

---

## [0.3.0] — chutes-miner-gpu chart · 2026-05-23

### Changed
- Registry nginx config replaced per-validator `map` routing with a single
  unconditional `set $upstream_host` directive, removing the dependency on the
  validator hotkey in the `Host` header. Both old and new VM hostnames route
  correctly with no fallback logic needed.
- Registry `NodePort` Service consolidated from one-per-validator
  (`registry-<hotkey>`) to a single static Service named `chutes-registry`.

---

[Unreleased]: https://github.com/chutesai/chutes-miner/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/chutesai/chutes-miner/compare/v0.4.7...v0.5.0
[0.3.0]: https://github.com/chutesai/chutes-miner/compare/chart-0.2.7...chart-0.3.0
