from unittest.mock import AsyncMock

import pytest
import yaml

from constants import CHUTE_ID, CHUTE_NAME, GPU_COUNT, SERVER_ID, SERVER_NAME

MOCK_AGENT_KUBECONFIG = yaml.safe_dump(
    {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [
            {
                "name": "test-node",
                "cluster": {
                    "server": "https://test-node:6443",
                    "certificate-authority-data": "TESTDATA",
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
                    "token": "test-token",
                },
            }
        ],
        "current-context": "test-node",
    },
    sort_keys=False,
)

@pytest.fixture
def mock_purge_deployments_response():
    """Mock response from the API."""
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(
        return_value={
            "status": "initiated",
            "deployments_purged": [
                {
                    "chute_id": CHUTE_ID,
                    "chute_name": CHUTE_NAME,
                    "server_id": SERVER_ID,
                    "server_name": SERVER_NAME,
                    "gpu_count": GPU_COUNT,
                }
            ],
        }
    )
    return mock_resp


@pytest.fixture
def mock_purge_deployment_response():
    """Mock response from the API."""
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(
        return_value={
            "status": "initiated",
            "deployment_purged": {
                "chute_id": CHUTE_ID,
                "chute_name": CHUTE_NAME,
                "server_id": SERVER_ID,
                "server_name": SERVER_NAME,
                "gpu_count": GPU_COUNT,
            },
        }
    )
    return mock_resp


@pytest.fixture
def mock_agent_kubeconfig_response():
    """Mock GET /config/kubeconfig response from agent."""
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"kubeconfig": MOCK_AGENT_KUBECONFIG})
    return mock_resp