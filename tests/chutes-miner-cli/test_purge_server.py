import json
import asyncio
from unittest.mock import patch, MagicMock
from chutes_miner_cli.cli import purge_server
from constants import CHUTE_NAME, GPU_COUNT, SERVER_ID, SERVER_NAME


def test_purge_server_by_name(
    mock_hotkey_content,
    mock_client_session,
    mock_purge_server_response,
    tmp_path,
    monkeypatch,
    capsys,
):
    """Purge all deployments from a server identified by name."""
    hotkey_file = tmp_path / "hotkey.json"
    hotkey_file.write_text(mock_hotkey_content)

    original_run = asyncio.run

    def mock_run(coro):
        return original_run(coro)

    monkeypatch.setattr(asyncio, "run", mock_run)
    _session = mock_client_session(mock_purge_server_response)
    monkeypatch.setattr("aiohttp.ClientSession", MagicMock(return_value=_session))

    with (
        patch("builtins.open", create=True) as mock_open,
        patch("chutes_miner_cli.cli.delete_preflight") as mock_preflight,
    ):
        mock_preflight.return_value = True
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock()
        mock_open.return_value.read = MagicMock(return_value=mock_hotkey_content)

        purge_server(
            name=SERVER_NAME,
            hotkey=str(hotkey_file),
            miner_api="http://test-miner-api:32000",
        )

    # Verify preflight was called with the server name.
    mock_preflight.assert_called_once_with(
        SERVER_NAME, str(hotkey_file), "http://test-miner-api:32000"
    )

    # Verify the DELETE request targeted the correct endpoint.
    _session.delete.assert_called_once()
    call_url = _session.delete.call_args[0][0]
    assert call_url == f"http://test-miner-api:32000/servers/{SERVER_NAME}/deployments"

    # Verify the JSON response was printed.
    captured = capsys.readouterr()
    output_json = json.loads(captured.out)
    assert output_json["status"] == "initiated"
    assert len(output_json["deployments_purged"]) == 1
    assert output_json["deployments_purged"][0]["chute_name"] == CHUTE_NAME
    assert output_json["deployments_purged"][0]["gpu_count"] == GPU_COUNT
    assert output_json["deployments_purged"][0]["server_name"] == SERVER_NAME


def test_purge_server_by_id(
    mock_hotkey_content,
    mock_client_session,
    mock_purge_server_response,
    tmp_path,
    monkeypatch,
    capsys,
):
    """Purge all deployments from a server identified by server ID."""
    hotkey_file = tmp_path / "hotkey.json"
    hotkey_file.write_text(mock_hotkey_content)

    original_run = asyncio.run

    def mock_run(coro):
        return original_run(coro)

    monkeypatch.setattr(asyncio, "run", mock_run)
    _session = mock_client_session(mock_purge_server_response)
    monkeypatch.setattr("aiohttp.ClientSession", MagicMock(return_value=_session))

    with (
        patch("builtins.open", create=True) as mock_open,
        patch("chutes_miner_cli.cli.delete_preflight") as mock_preflight,
    ):
        mock_preflight.return_value = True
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock()
        mock_open.return_value.read = MagicMock(return_value=mock_hotkey_content)

        purge_server(
            name=SERVER_ID,
            hotkey=str(hotkey_file),
            miner_api="http://test-miner-api:32000",
        )

    # Verify preflight was called with the server ID.
    mock_preflight.assert_called_once_with(
        SERVER_ID, str(hotkey_file), "http://test-miner-api:32000"
    )

    # Verify the DELETE request targeted the correct endpoint using the ID.
    _session.delete.assert_called_once()
    call_url = _session.delete.call_args[0][0]
    assert call_url == f"http://test-miner-api:32000/servers/{SERVER_ID}/deployments"

    # Verify the JSON response was printed.
    captured = capsys.readouterr()
    output_json = json.loads(captured.out)
    assert output_json["status"] == "initiated"
    assert len(output_json["deployments_purged"]) == 1
    assert output_json["deployments_purged"][0]["server_id"] == SERVER_ID


def test_purge_server_preflight_denied(
    mock_hotkey_content,
    mock_client_session,
    mock_purge_server_response,
    tmp_path,
    monkeypatch,
    capsys,
):
    """When preflight returns False (user aborts), no DELETE request is made."""
    hotkey_file = tmp_path / "hotkey.json"
    hotkey_file.write_text(mock_hotkey_content)

    original_run = asyncio.run

    def mock_run(coro):
        return original_run(coro)

    monkeypatch.setattr(asyncio, "run", mock_run)
    _session = mock_client_session(mock_purge_server_response)
    monkeypatch.setattr("aiohttp.ClientSession", MagicMock(return_value=_session))

    with (
        patch("builtins.open", create=True) as mock_open,
        patch("chutes_miner_cli.cli.delete_preflight") as mock_preflight,
    ):
        mock_preflight.return_value = False
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock()
        mock_open.return_value.read = MagicMock(return_value=mock_hotkey_content)

        purge_server(
            name=SERVER_NAME,
            hotkey=str(hotkey_file),
            miner_api="http://test-miner-api:32000",
        )

    # Preflight was called but returned False — DELETE must not be issued.
    mock_preflight.assert_called_once()
    _session.delete.assert_not_called()

    # Nothing should be printed.
    captured = capsys.readouterr()
    assert captured.out == ""


def test_purge_server_cli_integration(monkeypatch):
    """The purge-server command is registered and reachable via the CLI."""
    from chutes_miner_cli.cli import app
    from typer.testing import CliRunner

    runner = CliRunner()

    mock_run = MagicMock()
    monkeypatch.setattr(asyncio, "run", mock_run)

    result = runner.invoke(
        app,
        ["purge-server", "--name", SERVER_NAME, "--hotkey", "/path/to/hotkey.json"],
    )

    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_purge_server_missing_name_cli(monkeypatch):
    """Omitting --name must produce a clear CLI error and non-zero exit code."""
    from chutes_miner_cli.cli import app
    from typer.testing import CliRunner

    runner = CliRunner()

    result = runner.invoke(
        app,
        ["purge-server", "--hotkey", "/path/to/hotkey.json"],
    )

    assert result.exit_code != 0
