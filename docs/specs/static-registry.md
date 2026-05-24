# Cross-Repo Spec: chutes-miner — Static Registry Hostname

**Date**: 2026-05-23
**Status**: implemented
**Coordinating repo**: sek8s

---

## Context

sek8s is eliminating the hard-coded validator hotkey from TEE VM images. The registry hostname is changing from `${VALIDATOR_SS58}.localregistry.chutes.ai` to a static `localregistry.chutes.ai`.

The chutes-miner chart runs an nginx reverse proxy inside the miner's k3s cluster that routes image pull requests from the VM to the upstream registry. It currently uses the validator's hotkey in the hostname for routing via an nginx `map` directive.

### Why this matters

- The validator hotkey in the hostname serves no security purpose — it was originally adopted possibly for multi-validator support, which is not used in production
- Hotkey rotation under the current scheme invalidates all cosign signatures (cosign stores signatures keyed to the full image reference including hostname)
- Decoupling the hostname from any identity makes the system resilient to key rotation

---

## Change: registry-cm.yaml

**File**: `charts/chutes-miner-gpu/templates/registry-cm.yaml`

The nginx config currently uses the validator hotkey in the hostname routing:

```nginx
map $http_host $upstream_host {
    default {{ .Values.validators.defaultRegistry }};
    {{ .hotkey | lower }}.localregistry.chutes.ai {{ .Values.validators.defaultRegistry }};
}
```

(The exact template syntax may vary — the key point is that `.hotkey` or `.Values.hotkey` is used to construct a per-validator hostname.)

### ~~Option A: Update the map to use the static hostname~~

### Option B: Remove the map entirely ✓ **Selected**

Since there is only one validator per miner, the map is unnecessary. The `set` directive is unconditional — it doesn't depend on the `Host` header at all, so old VMs sending `<hotkey>.localregistry.chutes.ai` and new VMs sending `localregistry.chutes.ai` both route correctly with no fallback logic needed:

```nginx
set $upstream_host {{ .Values.validators.defaultRegistry }};
```

Also remove the now-unnecessary `map_hash_bucket_size` and `map_hash_max_size` directives.

### Other changes

- `registry-svc.yaml` previously created one `NodePort` Service per validator named `registry-<hotkey|lower>`. This was replaced with a single statically-named Service (`chutes-registry`).
- `X-Chutes-Hotkey` auth headers, and the `validators` JSON env var passed to the auth and agent containers, are **not** hostname-related and were left unchanged.
- `values.yaml` was not changed — the `hotkey` field in `validators.supported` is still consumed by auth/agent containers via the JSON env var.

---

## Deployment Ordering

This must be deployed at the same time as (or just before) the sek8s VM image update:

1. API changes deploy first (forge signs with new hostname, re-sign existing images)
2. **Miner chart + sek8s VM image deploy together** — both expect `localregistry.chutes.ai`

If the miner chart deploys before the VM update: old VMs still send requests with the old hostname, but since `$upstream_host` is set unconditionally, it routes correctly regardless.

If the VM update deploys before the miner chart: new VMs send requests with `localregistry.chutes.ai`, which also routes correctly.

The ordering is flexible in either direction — the static `set $upstream_host` does not inspect the `Host` header, so no fallback rule is needed.

---

## Failure Conditions

- ~~Removing the map breaks the `default` fallback for old VMs still using the hotkey-based hostname~~ — not applicable with Option B; the upstream is unconditional
- Chart values changes break existing Helm releases (use `--reuse-values` compatibility)
- Other parts of the chart still construct `${hotkey}.localregistry.chutes.ai` patterns that weren't updated
