## Chutes Miner CLI

This CLI ships with helpers for managing kubeconfigs when operating miner clusters. This document is the single source of truth for those workflows—other READMEs simply link here to avoid drift.

### Quick reference

| Question | Answer |
| --- | --- |
| Where do the commands run? | On the machine where you execute `chutes-miner-cli`. Nothing is copied anywhere else automatically. |
| Default output path | `~/.kube/chutes.config` unless you pass `--path`. |
| How do I point `kubectl` at it? | `export KUBECONFIG=~/.kube/chutes.config` or `kubectl --kubeconfig ~/.kube/chutes.config ...`. |
| Can I push it to another host? | Yes, but you must copy it yourself (example below). |

---

## `sync-kubeconfig`

Fetches the merged kubeconfig for **all** nodes that have already been registered with the miner API.

```bash
chutes-miner-cli sync-kubeconfig \
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
chutes-miner-cli sync-node-kubeconfig \
	--agent-api https://10.0.0.5:8443 \
	--context-name chutes-miner-gpu-0 \
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

## Verification checklist

- `chutes-miner-cli sync-kubeconfig ...` or `sync-node-kubeconfig ...` exits successfully.
- `stat ~/.kube/chutes.config` shows a recent timestamp.
- `KUBECONFIG` (or `--kubeconfig`) points to the path you just wrote.
- `kubectl config get-contexts` lists the expected contexts (control plane + all tracked nodes, plus any manual additions).
- Optional: run `sync_control_kubeconfig` to push the file to servers that need it.

