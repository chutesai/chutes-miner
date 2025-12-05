# `sync-kubeconfig` Reference

This guide explains exactly what the `chutes-miner-cli sync-kubeconfig` command does, where the resulting file is written, and how to move that kubeconfig to another machine (for example, your control node) when needed.

## TL;DR

| Question | Answer |
| --- | --- |
| **Where does the command run?** | On the same machine where you execute `chutes-miner-cli` (your laptop, bastion, or node). Nothing happens on the control node unless you run it **there**. |
| **Where does it write?** | `~/.kube/chutes.config` by default. If your `KUBECONFIG` env var points elsewhere, you will not see any changes until you point it (or `kubectl --kubeconfig`) at this file. |
| **Can it update kubeconfig on another server?** | No. Use `scp`/`rsync` or the helper function below to copy the generated file wherever you need it. |

---

## What the command actually does

```bash
chutes-miner-cli sync-kubeconfig \
  --hotkey ~/.bittensor/wallets/<wallet>/hotkeys/<hotkey> \
  --miner-api http://127.0.0.1:32000 \
  --path ~/.kube/chutes.config    # optional, defaults to this value
```

1. Authenticates to the miner API using your hotkey.
2. Calls `GET /servers/kubeconfig` on the miner API to request a merged kubeconfig that contains every registered cluster plus the control-plane context.
3. Writes the JSON/YAML response **to the filesystem on the machine running the CLI**. No remote copies occur.
4. Creates parent directories as needed and overwrites the file specified via `--path` (defaults to `~/.kube/chutes.config`).

### Interacting with the synced file

```bash
export KUBECONFIG=~/.kube/chutes.config
kubectl config get-contexts
kubectl --namespace chutes get pods
```

If you already have `KUBECONFIG` pointing somewhere else (e.g., `~/.kube/config`), either update the variable or pass `--kubeconfig ~/.kube/chutes.config` to each `kubectl`/`helm` command. Otherwise, it will look like nothing changed even though the file was updated.

## Common misconceptions cleared up

1. **"I ran the command locally; why doesn’t the control node have the new kubeconfig?"**  Because the CLI only touches the local filesystem. To update the control node you must copy the file manually.
2. **"kubectl still uses my old contexts."** Check `echo $KUBECONFIG`. If it points to a different file, either change it or merge the new file into your existing config.
3. **"Does this modify the nodes directly?"** No. It only downloads the aggregated kubeconfig from the miner API.

## Copying the kubeconfig to another machine

If you want to run the CLI on your laptop but maintain your operational kubeconfig on the control node (or any remote host), copy the file after syncing. Here’s a helper function you can drop into `~/.bashrc` or `~/.zshrc`:

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
sync_control_kubeconfig                 # uses defaults shown above
sync_control_kubeconfig ~/.kube/chutes.config admin my-control-node ~/.kube/chutes.config
```

This function:
1. Verifies the local file exists.
2. Uses `scp` to transfer it to the remote host.
3. Fixes permissions to `600` and shows the contexts on the remote side so you can confirm the copy worked.

Feel free to replace `scp` with `rsync` or add SSH options (port, identity file, etc.) to suit your environment.

## Verification checklist

- `chutes-miner-cli sync-kubeconfig ...` succeeds.
- `stat ~/.kube/chutes.config` shows a recent timestamp.
- `KUBECONFIG` includes the path (or you pass `--kubeconfig`).
- `kubectl config get-contexts` lists one context per registered server plus the control plane.
- Optional: run `sync_control_kubeconfig` to push the file to any server that needs it.

With these steps, miners always know exactly where the kubeconfig lives and how to shuttle it between machines without surprises.
