import yaml
import pytest
import typer
from unittest.mock import MagicMock

from chutes_miner_cli.cli import sync_node_kubeconfig


def _write_initial_kubeconfig(path: str, server_value: str):
    config = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [
            {
                "name": "test-node",
                "cluster": {
                    "server": server_value,
                    "certificate-authority-data": "LOCAL",
                },
            }
        ],
        "contexts": [
            {
                "name": "test-node",
                "context": {
                    "cluster": "test-node",
                    "user": "test-node-user",
                    "namespace": "chutes",
                },
            }
        ],
        "users": [
            {
                "name": "test-node-user",
                "user": {
                    "token": "local-token",
                },
            }
        ],
        "current-context": "test-node",
    }
    with open(path, "w") as fh:
        yaml.safe_dump(config, fh)


def test_sync_node_kubeconfig_adds_context(
    mock_hotkey_content,
    mock_get_client_session,
    mock_agent_kubeconfig_response,
    tmp_path,
    monkeypatch,
):
    hotkey_file = tmp_path / "hotkey.json"
    hotkey_file.write_text(mock_hotkey_content)
    kubeconfig_path = tmp_path / "merged.config"

    session = mock_get_client_session(mock_agent_kubeconfig_response)
    monkeypatch.setattr("aiohttp.ClientSession", MagicMock(return_value=session))

    sync_node_kubeconfig(
        agent_api="https://test-agent",
        context_name="test-node",
        path=str(kubeconfig_path),
        hotkey=str(hotkey_file),
    )

    merged = yaml.safe_load(kubeconfig_path.read_text())
    assert merged["current-context"] == "test-node"
    assert any(ctx["name"] == "test-node" for ctx in merged["contexts"])
    assert any(
        cluster["cluster"]["server"] == "https://test-node:6443"
        for cluster in merged["clusters"]
    )


def test_sync_node_kubeconfig_requires_overwrite(
    mock_hotkey_content,
    mock_get_client_session,
    mock_agent_kubeconfig_response,
    tmp_path,
    monkeypatch,
):
    hotkey_file = tmp_path / "hotkey.json"
    hotkey_file.write_text(mock_hotkey_content)
    kubeconfig_path = tmp_path / "merged.config"
    _write_initial_kubeconfig(str(kubeconfig_path), "https://old-server")

    session = mock_get_client_session(mock_agent_kubeconfig_response)
    monkeypatch.setattr("aiohttp.ClientSession", MagicMock(return_value=session))

    with pytest.raises(typer.Exit) as excinfo:
        sync_node_kubeconfig(
            agent_api="https://test-agent",
            context_name="test-node",
            path=str(kubeconfig_path),
            hotkey=str(hotkey_file),
            overwrite=False,
        )

    assert excinfo.value.exit_code == 1
    existing = yaml.safe_load(kubeconfig_path.read_text())
    assert existing["clusters"][0]["cluster"]["server"] == "https://old-server"


def test_sync_node_kubeconfig_overwrites_when_requested(
    mock_hotkey_content,
    mock_get_client_session,
    mock_agent_kubeconfig_response,
    tmp_path,
    monkeypatch,
):
    hotkey_file = tmp_path / "hotkey.json"
    hotkey_file.write_text(mock_hotkey_content)
    kubeconfig_path = tmp_path / "merged.config"
    _write_initial_kubeconfig(str(kubeconfig_path), "https://old-server")

    session = mock_get_client_session(mock_agent_kubeconfig_response)
    monkeypatch.setattr("aiohttp.ClientSession", MagicMock(return_value=session))

    sync_node_kubeconfig(
        agent_api="https://test-agent",
        context_name="test-node",
        path=str(kubeconfig_path),
        hotkey=str(hotkey_file),
        overwrite=True,
    )

    merged = yaml.safe_load(kubeconfig_path.read_text())
    assert merged["clusters"][0]["cluster"]["server"] == "https://test-node:6443"
    assert merged["users"][0]["user"]["token"] == "test-token"