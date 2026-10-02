"""
Tests for the TEE maintenance commands (pending -> maintenance flow).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from chutes_miner_cli.cli import app

SERVER_NAME = "tee-vm-1"
SERVER_ID = "server-abc-123"
VALIDATOR_API = "http://test-validator"
MINER_API = "http://test-miner-api:32000"
PENDING_DEADLINE = "2026-10-02T15:00:00+00:00"
SURVIVOR = {"chute_id": "chute-sole", "instance_id": "inst-sole"}
WINDOW = {
    "id": "window-1",
    "target_measurement_version": "0.4.0",
    "upgrade_window_start": "2026-10-01 00:00:00+00:00",
    "upgrade_window_end": "2026-10-08 00:00:00+00:00",
    "max_concurrent_per_miner": 1,
}

runner = CliRunner()


def _response(payload, status=200):
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=payload)
    resp.text = AsyncMock(return_value=json.dumps(payload))
    return resp


def _cm(resp):
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


class FakeSession:
    """aiohttp.ClientSession stand-in that serves canned responses by (method, url)."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    def _request(self, method, url, **kwargs):
        self.calls.append((method, url))
        return _cm(self.routes[(method, url)])

    def get(self, url, **kwargs):
        return self._request("GET", url, **kwargs)

    def put(self, url, **kwargs):
        return self._request("PUT", url, **kwargs)


@pytest.fixture
def fake_session():
    """Patch aiohttp.ClientSession (and request signing) with a routed fake."""

    def _install(routes):
        session = FakeSession(routes)
        patcher_session = patch("aiohttp.ClientSession", MagicMock(return_value=session))
        patcher_sign = patch(
            "chutes_miner_cli.tee_maintenance.sign_request", return_value=({}, None)
        )
        # Wide console so Rich doesn't wrap/truncate the text the tests assert on.
        patcher_console = patch(
            "chutes_miner_cli.tee_maintenance.console", Console(width=300, no_color=True)
        )
        _install.patchers = [patcher_session, patcher_sign, patcher_console]
        for p in _install.patchers:
            p.start()
        return session

    yield _install
    for p in getattr(_install, "patchers", []):
        p.stop()


def _invoke(*args):
    return runner.invoke(
        app,
        ["tee", *args, "--hotkey", "hotkey.json", "--validator-api", VALIDATOR_API],
    )


def _start_maintenance_routes(preflight, confirm, confirm_status):
    return {
        ("GET", f"{VALIDATOR_API}/servers/{SERVER_NAME}/maintenance/preflight"): _response(
            preflight
        ),
        ("GET", f"{MINER_API}/servers/{SERVER_NAME}/lock"): _response(
            {"name": SERVER_NAME, "locked": True}
        ),
        ("PUT", f"{VALIDATOR_API}/servers/{SERVER_NAME}/maintenance"): _response(
            confirm, status=confirm_status
        ),
    }


def test_start_maintenance_pending_warns_not_to_reboot(fake_session):
    preflight = {"eligible": True, "current_slots": 0, "limit": 1, "sole_survivors": [SURVIVOR]}
    confirm = {
        "server_id": SERVER_ID,
        "maintenance_status": "pending",
        "purged_instance_ids": ["inst-other"],
        "awaiting_replacement": [SURVIVOR],
        "pending_deadline": PENDING_DEADLINE,
        "window": WINDOW,
    }
    session = fake_session(_start_maintenance_routes(preflight, confirm, 202))

    result = _invoke("start-maintenance", "--name", SERVER_NAME, "--yes", "--miner-api", MINER_API)

    assert result.exit_code == 0, result.output
    assert "sole survivors above" in result.output
    assert "Maintenance pending." in result.output
    assert "Do NOT reboot yet." in result.output
    assert "chute-sole" in result.output
    assert PENDING_DEADLINE in result.output
    assert f"chutes-miner tee maintenance-status --name {SERVER_NAME}" in result.output
    assert "Maintenance confirmed" not in result.output
    assert ("PUT", f"{VALIDATOR_API}/servers/{SERVER_NAME}/maintenance") in session.calls


def test_start_maintenance_without_survivors_is_safe_to_reboot(fake_session):
    preflight = {"eligible": True, "current_slots": 0, "limit": 1, "sole_survivors": []}
    confirm = {
        "server_id": SERVER_ID,
        "maintenance_status": "maintenance",
        "purged_instance_ids": ["inst-1"],
        "window": WINDOW,
    }
    fake_session(_start_maintenance_routes(preflight, confirm, 200))

    result = _invoke("start-maintenance", "--name", SERVER_NAME, "--yes", "--miner-api", MINER_API)

    assert result.exit_code == 0, result.output
    assert "Maintenance confirmed; safe to reboot." in result.output
    assert "Do NOT reboot yet." not in result.output


def test_start_maintenance_raw_json_prints_only_json(fake_session):
    preflight = {"eligible": True, "current_slots": 0, "limit": 1, "sole_survivors": []}
    confirm = {"server_id": SERVER_ID, "maintenance_status": "maintenance", "window": WINDOW}
    fake_session(_start_maintenance_routes(preflight, confirm, 200))

    result = _invoke(
        "start-maintenance", "--name", SERVER_NAME, "--yes", "--raw-json", "--miner-api", MINER_API
    )

    assert result.exit_code == 0, result.output
    json_start = result.output.index("{")
    assert json.loads(result.output[json_start:]) == confirm


def test_start_maintenance_denied_shows_reason(fake_session):
    preflight = {
        "eligible": False,
        "current_slots": 1,
        "limit": 1,
        "denial_reasons": [{"reason": "maintenance_requested"}],
        "sole_survivors": [],
    }
    session = fake_session(_start_maintenance_routes(preflight, {}, 200))

    result = _invoke("start-maintenance", "--name", SERVER_NAME, "--yes", "--miner-api", MINER_API)

    assert result.exit_code == 1
    assert "maintenance_requested" in result.output
    assert not any(method == "PUT" for method, _ in session.calls)


@pytest.mark.parametrize(
    "status,expected,unexpected",
    [
        ("pending", "Do NOT reboot yet.", "Safe to reboot"),
        ("maintenance", "Safe to reboot into the upgrade.", "Do NOT reboot yet."),
        ("none", "Server is not in maintenance.", "Safe to reboot"),
    ],
)
def test_maintenance_status_for_server(fake_session, status, expected, unexpected):
    payload = {
        "server_id": SERVER_ID,
        "maintenance_status": status,
        "awaiting_replacement": [SURVIVOR] if status == "pending" else [],
        "pending_deadline": PENDING_DEADLINE if status == "pending" else None,
        "reconciled_at": "2026-10-02T12:00:00+00:00",
    }
    session = fake_session(
        {("GET", f"{VALIDATOR_API}/servers/{SERVER_NAME}/maintenance"): _response(payload)}
    )

    result = _invoke("maintenance-status", "--name", SERVER_NAME)

    assert result.exit_code == 0, result.output
    assert expected in result.output
    assert unexpected not in result.output
    assert session.calls == [("GET", f"{VALIDATOR_API}/servers/{SERVER_NAME}/maintenance")]


def test_maintenance_status_policy_shows_server_status(fake_session):
    payload = {
        "active_window": WINDOW,
        "window_open": True,
        "current_slots": 1,
        "servers": [
            {
                "server_id": SERVER_ID,
                "name": SERVER_NAME,
                "version": "0.3.0",
                "needs_upgrade": True,
                "in_maintenance": False,
                "maintenance_status": "pending",
            }
        ],
    }
    session = fake_session(
        {("GET", f"{VALIDATOR_API}/servers/maintenance/policy"): _response(payload)}
    )

    result = _invoke("maintenance-status")

    assert result.exit_code == 0, result.output
    assert "Pending" in result.output
    assert session.calls == [("GET", f"{VALIDATOR_API}/servers/maintenance/policy")]
