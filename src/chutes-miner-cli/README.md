## Chutes Miner CLI

This CLI ships with helpers for managing kubeconfigs when operating miner clusters, plus streaming pre-activation chute logs via the validator API. This document is the single source of truth for those workflows—other READMEs simply link here to avoid drift.

### Quick reference


| Question                        | Answer                                                                                          |
| ------------------------------- | ----------------------------------------------------------------------------------------------- |
| Where do the commands run?      | On the machine where you execute `chutes-miner`. Nothing is copied anywhere else automatically. |
| Default output path             | `~/.kube/chutes.config` unless you pass `--path`.                                               |
| How do I point `kubectl` at it? | `export KUBECONFIG=~/.kube/chutes.config` or `kubectl --kubeconfig ~/.kube/chutes.config ...`.  |
| Can I push it to another host?  | Yes, but you must copy it yourself (example below).                                             |


---

## `sync-kubeconfig`

Fetches the merged kubeconfig for **all** nodes that have already been registered with the miner API.

```bash
chutes-miner sync-kubeconfig \
	--hotkey ~/.bittensor/wallets/<wallet>/hotkeys/<hotkey>.json \
	--miner-api http://127.0.0.1:32000 \
	--path ~/.kube/chutes.config   # optional, defaults to this value
```

1. Signs the request with your miner hotkey.
2. Calls `GET /servers/kubeconfig` on the miner API.
3. Writes the returned kubeconfig to the local filesystem, creating parent directories as needed and overwriting the target file.

After syncing:

```bash
export KUBECONFIG=~/.kube/chutes.config
kubectl config get-contexts
kubectl --namespace chutes get pods
```

> **Important:** `KUBECONFIG` must include the path you wrote to, otherwise `kubectl` keeps using whatever file it was already pointed at.

## `sync-node-kubeconfig`

Fetches a **single** context directly from a node before it has been added to the miner database. The CLI talks to the agent on that node at `/config/kubeconfig`, extracts the requested context, and merges it into your local kubeconfig.

```bash
chutes-miner sync-node-kubeconfig \
	--agent-api https://10.0.0.5:8443 \
	--context-name chutes-miner-gpu-0 \   # Set this to the name of your node
	--hotkey ~/.bittensor/wallets/<wallet>/hotkeys/<hotkey>.json \
	--path ~/.kube/chutes.config \
	--overwrite               # optional, required if the context already exists
```

Behavior highlights:

- Requires the same signed request headers as the miner API (`purpose="registration"`, management mode).
- Parses the returned kubeconfig, finds the specified context, and copies only the context/cluster/user bundle.
- Refuses to replace existing entries unless `--overwrite` is provided, which helps prevent accidental credential swaps.

## Copying the kubeconfig elsewhere

Both commands only touch the local filesystem. To make the synced kubeconfig available on another host, copy it manually. Example helper function:

```bash
sync_control_kubeconfig() {
	local local_cfg=${1:-$HOME/.kube/chutes.config}
	local remote_user=${2:-ubuntu}
	local remote_host=${3:-chutes-miner-cpu-0}
	local remote_path=${4:-.kube/chutes.config}

	if [ ! -f "$local_cfg" ]; then
		echo "Local kubeconfig not found at $local_cfg" >&2
		return 1
	fi

	echo "Copying $local_cfg to $remote_user@$remote_host:$remote_path"
	scp "$local_cfg" "$remote_user@$remote_host:$remote_path"
	ssh "$remote_user@$remote_host" "chmod 600 $remote_path && export KUBECONFIG=$remote_path && kubectl config get-contexts"
}
```

Usage:

```bash
sync_control_kubeconfig                             # uses defaults above
sync_control_kubeconfig ~/.kube/chutes.config admin my-control-node ~/.kube/chutes.config
```

Feel free to swap `scp` for `rsync`, add SSH options, or integrate with your automation tooling.

## `instance-logs`

Streams startup logs for a **public** chute instance (via the validator `GET /miner/instance_logs` endpoint) using the launch JWT. Pass `--jwt` or set `CHUTES_LAUNCH_JWT`.

On a miner cluster, the chute pod’s `CHUTES_LAUNCH_JWT` env var holds that token. This helper takes the **pod name** as the first argument (namespace `chutes`):

```bash
get_chute_jwt() {
	kubectl get po "$1" -n chutes -o json \
		| jq -r '.spec.containers[0].env[] | select(.name == "CHUTES_LAUNCH_JWT") | .value'
}
```

Requires `kubectl` (with a context that can see the pod) and `[jq](https://jqlang.github.io/jq/)`.

Example:

```bash
export CHUTES_LAUNCH_JWT="$(get_chute_jwt chute-2234db67-b453-4198-8ece-74f1f8ea3d03-58pph)"
chutes-miner instance-logs
```

## Verification checklist

- `chutes-miner sync-kubeconfig ...` or `sync-node-kubeconfig ...` exits successfully.
- `stat ~/.kube/chutes.config` shows a recent timestamp.
- `KUBECONFIG` (or `--kubeconfig`) points to the path you just wrote.
- `kubectl config get-contexts` lists the expected contexts (control plane + all tracked nodes, plus any manual additions).
- Optional: run `sync_control_kubeconfig` to push the file to servers that need it.
- Optional: `chutes-miner instance-logs` with a JWT from `get_chute_jwt` (see above) streams logs until the validator closes the stream or you interrupt; use the stderr resume hint and `--cursor` to continue after errors or Ctrl+C.

