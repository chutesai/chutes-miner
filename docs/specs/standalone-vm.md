# Feature Spec: Standalone VM

**Date**: 2026-03-16
**Status**: in progress

---

## Context

- **Packages affected**: ansible/k3s, charts/chutes-control (rename from chutes-miner), charts/chutes-gpu (rename from chutes-miner-gpu), charts/chutes-executor (new), src/chutes_miner
- **Key files**:
  - `src/chutes-miner/chutes_miner/gepetto.py` -- current scheduler/reconciler (~2200 lines)
  - `src/chutes-miner/chutes_miner/api/server/router.py` -- server registration (add-node)
  - `src/chutes-miner/chutes_miner/api/server/verification.py` -- GraVal/TEE verification, hardcoded port 30443
  - `src/chutes-miner/chutes_miner/api/k8s/operator.py` -- SingleCluster/MultiCluster K8s operators
  - `src/chutes-miner/chutes_miner/api/socket_client.py` -- WebSocket connection to validator, publishes to Redis
  - `src/chutes-miner/chutes_miner/api/redis_pubsub.py` -- Redis pubsub listener, dispatches events to handlers
  - `src/chutes-agent/chutes_agent/monitor.py` -- agent resource monitoring, reports to control plane
- **Dependencies**: Will require corresponding updates in chutes-api [[https://github.com/chutesai/chutes-api]](https://github.com/chutesai/chutes-api]) to accommodate management of a single node instead of multiple clusters managed via a central control node for scheduling. The API changes should be minimal and consist of providing an optional query parameter for the cluster name. If the cluster name is included the assumption is we are limiting remote inventory and queries to resources specific to this cluster, if not included it will be limited to resources for the entire miner hotkey. TEE VM guest-side registration and fail-fast logic lives in sek8s [[https://github.com/chutesai/sek8s]](https://github.com/chutesai/sek8s]).

---

## Design Decisions

1. **Standalone scheduler as a new module (gepetto-lite)**: Create a standalone scheduler module with full feature parity (reconciliation, autoscaling, preemption, activator, pubsub). Long-term, extract core scheduling/reconciliation logic from Gepetto into shared modules that both consume, letting miners customize only event-response behavior.
2. **Cluster-scoped API reconciliation**: The standalone scheduler queries the chutes-api with an optional `cluster_name` parameter to scope reconciliation to resources for this specific cluster under the configured hotkey. This is the key behavioral difference from Gepetto, which reconciles all clusters for the hotkey.
3. **Control-node-as-GPU (Scenario 3) via local agent**: Run chutes-agent on the control node, reporting to chutes-monitor which writes resource state to Redis -- same data path as remote clusters. `MultiClusterK8sOperator` detects when the target cluster matches the local node and uses in-cluster kubeconfig. The server is still registered in the DB for CLI inventory and scheduling.
4. **Separate chart for standalone (no Postgres, keep Redis)**: Create a new chart (`charts/chutes-executor/`). No Postgres needed -- deployment state comes from the cluster and remote API. Redis is kept for the SocketClient -> pubsub -> scheduler event pipeline (both `socket_client.py` and `redis_pubsub.py` require a real Redis connection). No local state persistence needed for registration; the validator inventory (`/miner/servers`) is the source of truth for server identity.
5. **Dual-path port configuration**:
  - Non-TEE: Ports configured via Helm values / Ansible vars. Miners provide custom ports in inventory, run playbooks, and services are created with the correct NodePorts.
  - TEE: Charts are pre-baked into the VM image and ports aren't known at build time. Ports must be configurable at first boot -- either by modifying static manifests, editing existing K8s manifests, or upgrading charts with custom values during first-boot provisioning.
6. **Port discovery via K8s service labels**: Rather than passing ports through config/env vars, the K8s operator queries services in each cluster by a well-known label to discover actual NodePort values. Works across all topologies -- the operator resolves ports for whichever cluster it's interacting with (self or remote). The cluster is the source of truth for its own ports.

---

## API Changes

- **No miner API deployment in standalone mode**: The standalone scheduler handles validator communication directly (WebSocket + HTTP). The only component from the existing API reused is the Redis pubsub event pipeline.
- **Standalone self-registration**: On startup, the standalone scheduler:
  1. Reads local config (name, GPU short ref, hotkey, IP, ports).
  2. Queries `/miner/servers` on the validator to check if the server already exists with matching config (name, IP, ports).
  3. If match found: skip registration, proceed to normal reconciliation/scheduling.
  4. If not found: run full registration (verification + advertise to validator).
  5. If conflict (exists but config differs, or 409): **fail fast and shut down**. Operator must investigate. This avoids accidentally removing an active VM due to misconfiguration (duplicate name, duplicate ports). Smarter auto-reconciliation can be added later.
- **Registration module**: Extract registration logic from the miner API servers endpoint into a shared module so both the existing add-node flow and the standalone registration API can consume it without duplication.
- **Lightweight registration API**: A small API endpoint that accepts hotkey, server name, and GPU config from the VM guest (sek8s). Returns success/conflict/error so the guest can handle fail-fast shutdown (k3s cannot shut down the VM from inside the cluster). Guest-side logic lives in sek8s.
- **chutes-api changes** (external repo): Add optional `cluster_name` query parameter to miner-facing endpoints. Scopes responses to resources for that cluster under the hotkey. Omitting preserves existing behavior. Minimal and backward-compatible.
- **Port discovery via K8s service labels**: K8s operator queries services by label to discover NodePort values per cluster. Works across all topologies.
- **NodeArgs / server advertisement**: Add `agent_port` and `attestation_port` fields with backward-compatible defaults (32000, 30443). No port range in the schema -- the scheduler knows its range internally; job ports come from the workload itself during verification with the validator.
- **No schema migrations**: No Postgres in standalone. Port fields on the server record are a chutes-api concern.

---

## Goal

The final implementation needs to support the following scenarios:

1. Deploying a chutes miner with a central control node. In this scenario we deploy the chutes-control chart to a central control node, then we deploy the chutes-gpu charts to all GPU clusters and the central control node handles scheduling workloads among all of the GPU clusters.
2. Deploying as part of a standalone VM. In this scenario we deploy the chutes-executor chart which includes Redis, gepetto-lite (standalone scheduler), and registry. Gepetto-lite handles deploying chutes within this cluster and reconciling state with the validator using the optional cluster parameter to limit reconciliation to resources specific to this cluster for the configured hotkey. Self-registration is idempotent, using the validator inventory as the source of truth. On conflict, the VM fails fast and shuts down.
3. Deploying a chutes miner where the central control node is also a GPU node and could deploy workloads to itself. This is the same setup as scenario 1, but requires running chutes-agent on the control node and having the MultiClusterK8sOperator detect the local cluster and use in-cluster kubeconfig. The server is registered in the DB for CLI inventory and scheduling.

Additional features that need to be supported:

1. Configurable ports for agent (currently 32000) and attestation proxy (currently 30443) to support multiple servers behind NAT. Port discovery is done via K8s service labels so the operator resolves ports per-cluster regardless of topology.
2. NAT port range isolation: each standalone VM behind NAT is assigned a predefined ephemeral port range (subset of 32000+ NodePort range). The standalone scheduler only allocates NodePorts within its assigned range so NAT routing works correctly.

---

## Constraints

- **Strictly additive**: Zero breaking changes to existing Scenario 1 (multi-cluster) behavior. All standalone/hybrid functionality is opt-in via configuration. Existing deployments work without modification.
- **Spec scope**: This spec covers chutes-miner repo changes and defines the API contract that chutes-api must implement. chutes-api and sek8s implementation details are separate work items.
- **Multi-GPU support**: Standalone VMs are typically multi-GPU machines (e.g., 8xH200). The scheduler must handle all GPUs on the node, not assume single-GPU.
- **NAT port range isolation**: Each standalone VM behind NAT gets a predefined ephemeral port range (subset of the 32000+ K8s NodePort range). The standalone scheduler only allocates NodePorts within its assigned range so the NAT provider can route traffic to the correct VM. This range is part of the VM's configuration alongside the agent and attestation ports.
- **Fail-fast on registration conflict**: If self-registration hits a conflict (409, config mismatch), the VM shuts down rather than attempting auto-remediation. Prevents accidental removal of active VMs due to misconfiguration.
- **Idempotent startup**: The standalone scheduler must safely handle repeated restarts/reboots. Validator inventory is the source of truth -- no local state persistence required beyond Helm/boot config.
- **Full scheduling feature parity**: The standalone scheduler supports reconciliation, autoscaling, preemption, activator, and pubsub -- same capabilities as Gepetto.
- **Async-first**: All network I/O (validator HTTP, WebSocket, K8s API) must use async patterns per AGENT.md.

---

## Output Format

1. **Registration module** -- Extract registration logic from the miner API servers endpoint (`src/chutes-miner/chutes_miner/api/server/`) into a shared module (e.g., `chutes_miner/registration/`). Covers inventory check against `/miner/servers`, verification (GraVal/TEE), and server advertisement. Consumed by the miner API (existing add-node flow) and the standalone registration API.
2. **Lightweight registration API** -- A small API endpoint that accepts hotkey, server name, and GPU config from the VM guest. Returns success/conflict/error so the guest (managed in sek8s) can handle fail-fast shutdown. Guest-side logic lives in sek8s, not this repo.
3. **Gepetto-lite** -- `src/chutes-miner/chutes_miner/gepetto_lite.py` (or `standalone/gepetto_lite.py`). Full scheduling feature parity: reconciliation, autoscaling, preemption, activator, pubsub. Cluster-scoped API queries via `cluster_name`. Port-range-aware NodePort allocation for NAT environments.
4. **K8s operator changes** -- `src/chutes-miner/chutes_miner/api/k8s/operator.py`: MultiClusterK8sOperator detects local cluster and uses in-cluster kubeconfig (Scenario 3). Port discovery via K8s service labels across all operator types.
5. **Helm chart** -- `charts/chutes-executor/`: Redis, gepetto-lite, registry. No Postgres, no agent (only needed for central control coordination), no attestation proxy (static manifest in VM). Service NodePorts set via Helm values (non-TEE) or via helm upgrade / manifest patching at first boot (TEE where ports aren't known at build time). ConfigMap provides the job NodePort range to gepetto-lite.
6. **NodeArgs / server advertisement contract** -- Add `agent_port` and `attestation_port` fields with backward-compatible defaults (32000, 30443). No port range in the schema -- scheduler knows its range internally; job ports come from the workload during verification.
7. **Verification module updates** -- `src/chutes-miner/chutes_miner/api/server/verification.py`: replace hardcoded port 30443 with value resolved from K8s service label discovery.

---

## Failure Conditions

- Existing multi-cluster deployments (Scenario 1) break or behave differently after these changes
- A standalone VM registration accidentally removes or overwrites an active server in the validator inventory
- Gepetto-lite allocates NodePorts outside the configured range, causing NAT routing failures for VMs behind NAT
- The registration API is not idempotent -- repeated calls with the same config produce errors or duplicate entries
- The K8s operator fails to resolve the correct kubeconfig when the control node is also a GPU node (Scenario 3), causing deployment failures
- Port discovery fails silently -- if a service label is missing, the system should error explicitly rather than fall back to a wrong port
- The chutes-api `cluster_name` parameter, when omitted, changes existing behavior for current miners
- Gepetto-lite loses validator events during restart/reconnect (same reliability expectations as Gepetto)
- The standalone chart requires Postgres to function

---

## Rollout Notes

- **Chart renames**: `charts/chutes-miner` -> `charts/chutes-control`, `charts/chutes-miner-gpu` -> `charts/chutes-gpu`. New chart `charts/chutes-executor/` ships alongside. Renames clarify topology roles (control plane, GPU worker, standalone executor).
- **chutes-api dependency**: The optional `cluster_name` query parameter must be deployed on the API side before standalone VMs can function. Existing miners are unaffected (parameter is optional).
- **NodeArgs defaults**: `agent_port=32000` and `attestation_port=30443` defaults ensure existing server advertisements are backward-compatible. No migration needed for current servers.
- **Scenario 3 (control as GPU)**: Opt-in. Requires running chutes-agent on the control node and labeling it appropriately. Existing control-only deployments are unaffected.
- **TEE VM integration**: Requires coordinated changes in sek8s for first-boot port configuration and guest-side registration/fail-fast logic. The chutes-miner side provides the registration API; sek8s consumes it.
- **Registration module refactor**: Extracting registration into a shared module changes internal code organization but no external behavior. Should be verified against existing add-node flows.
- **Port discovery rollout**: K8s service labels for port discovery need to be added to existing charts (`chutes-gpu` services) for the operator to resolve ports. This can be added as a non-breaking label addition.
- **No feature flags needed**: Standalone vs multi-cluster is determined by which chart is deployed, not runtime flags.

