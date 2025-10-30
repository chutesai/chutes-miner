from enum import auto
from typing import AsyncGenerator
from unittest.mock import Mock, patch
from chutes_common.k8s import WatchEvent, WatchEventType
from kubernetes.client import V1Node, V1ObjectMeta, V1JobList, V1Pod
from chutes_common.schemas.gpu import GPU
from chutes_common.schemas.server import Server
from chutes_miner.api.config import Settings
from chutes_miner.api.k8s.operator import K8sOperator
from unittest.mock import AsyncMock, MagicMock, Mock, _patch, patch
from typing import AsyncGenerator

import pytest

class MockDependencies:
    """Constants for all mocked dependencies in bootstrap_server"""
    K8S_OPERATOR = 'K8sOperator'
    GET_SESSION = 'get_session'
    SERVER = 'Server'
    GPU = 'GPU'
    UPDATE = 'update'
    SIGN_REQUEST = 'sign_request'
    STOP_SERVER_MONITORING = 'stop_server_monitoring'
    START_SERVER_MONITORING = 'start_server_monitoring'
    MULTI_CLUSTER_KUBE_CONFIG = 'MultiClusterKubeConfig'
    TRACK_SERVER = 'track_server'
    SETTINGS = 'settings'
    LOGGER = 'logger'
    SSE_MESSAGE = 'sse_message'

class MockStrategyDependencies:
    """Constants for all mocked dependencies in verification strategy"""
    K8S_OPERATOR = 'K8sOperator'
    GET_SESSION = 'get_session'
    SELECT = 'select'
    UPDATE = 'update'
    VALIDATOR_BY_HOTKEY = 'validator_by_hotkey'
    SIGN_REQUEST = 'sign_request'
    SETTINGS = 'settings'
    LOGGER = 'logger'
    SSE_MESSAGE = 'sse_message'


class MockServerArgs:
    """Mock ServerArgs class for testing"""
    def __init__(self, validator="test_validator", hourly_cost=1.0, 
                 gpu_short_ref="rtx4090", agent_api=None):
        self.validator = validator
        self.hourly_cost = hourly_cost
        self.gpu_short_ref = gpu_short_ref
        self.agent_api = agent_api


class MockKubeConfig:
    """Mock KubeConfig class for testing"""
    pass


class MockGPU:
    """Mock GPU object"""
    def __init__(self, gpu_id="gpu_123", device_info=None):
        self.gpu_id = gpu_id
        self.device_info = device_info or {"name": "RTX 4090"}


class MockServer:
    """Mock Server object"""
    def __init__(self, server_id="server_123", validator="test_validator", 
                 cpu_per_gpu=4, memory_per_gpu=16, gpus=None):
        self.server_id = server_id
        self.validator = validator
        self.cpu_per_gpu = cpu_per_gpu
        self.memory_per_gpu = memory_per_gpu
        self.gpus = gpus or []


class MockValidator:
    """Mock Validator object"""
    def __init__(self, hotkey="test_hotkey", api="http://test-api.com"):
        self.hotkey = hotkey
        self.api = api


class GraValBootstrapFailure(Exception):
    """Mock exception for testing"""
    pass


async def collect_sse_messages(async_gen: AsyncGenerator) -> list:
    """Helper to collect all SSE messages from the generator"""
    messages = []
    async for message in async_gen:
        messages.append(message)
    return messages

@pytest.fixture
def mock_node(mock_gpus):
    """Create a mock V1Node object"""
    node = V1Node()
    node.metadata = V1ObjectMeta()
    node.metadata.uid = "test-node-uid-123"
    node.metadata.name = "test-node-name"
    node.metadata.labels = {
        "nvidia.com/gpu.count": str(len(mock_gpus)),
        "gpu-short-ref": "rtx4090"
    }
    return node

@pytest.fixture
def mock_pod():
    pod = MagicMock(spec=V1Pod)
    pod.status.phase = "Running"
    pod.status.container_statuses = [
        MagicMock(ready=True)
    ]
    return pod

@pytest.fixture
def mock_server_args():
    """Create mock ServerArgs"""
    return MockServerArgs(
        validator="test_validator",
        hourly_cost=2.5,
        gpu_short_ref="rtx4090",
        agent_api=None
    )


@pytest.fixture
def mock_server_args_with_agent():
    """Create mock ServerArgs with agent API"""
    return MockServerArgs(
        validator="test_validator",
        hourly_cost=2.5,
        gpu_short_ref="rtx4090",
        agent_api="http://agent-api.com"
    )


@pytest.fixture
def mock_kubeconfig():
    """Create mock KubeConfig"""
    return MockKubeConfig()


@pytest.fixture
def mock_gpus():
    """Create mock GPU list"""
    return [
        MockGPU("gpu_1", {"name": "RTX 4090"}),
        MockGPU("gpu_2", {"name": "RTX 4090"}),
    ]


@pytest.fixture
def mock_gpu():
    return Mock(spec=GPU)

@pytest.fixture
def mock_server():
    """Create mock Server object"""
    _mock_server = Mock(spec=Server)
    _mock_server.server_id="test-node-uid-123",
    _mock_server.validator="test_validator",
    _mock_server.cpu_per_gpu=8,
    _mock_server.memory_per_gpu=32
    _mock_server.ip_address = "192.168.0.1"
    _mock_server.verification_prot = "32689"

    return _mock_server


@pytest.fixture
def mock_validator():
    """Create mock Validator object"""
    return MockValidator("test_hotkey", "http://test-validator.com")


@pytest.fixture
def mock_validator_nodes():
    """Create mock validator nodes response"""
    return [
        {"seed": "test_seed_123", "gpu_id": "gpu_1"},
        {"seed": "test_seed_123", "gpu_id": "gpu_2"},
    ]

@pytest.fixture
def mock_settings(mock_validator):
    _mock_settings = Mock(spec=Settings)
    _mock_settings.validators = [mock_validator]
    _mock_settings.nvidia_runtime = "nvidia"
    _mock_settings.graval_bootstrap_image = "graval:latest"
    _mock_settings.miner_ss58 = "abcd1234"
    _mock_settings.graval_bootstrap_timeout = 5

    return _mock_settings

@pytest.fixture
def mock_k8s_operator(mock_graval_service, mock_pod):
    _mock_operator = Mock(spec=K8sOperator)
    _mock_operator.return_value = _mock_operator
    _mock_operator.get_jobs.return_value = V1JobList(items=[])
    _mock_operator.deploy_graval.return_value = (Mock(), mock_graval_service)
    _mock_operator.watch_pods.side_effect = [
        [WatchEvent(type=WatchEventType.MODIFIED, object=mock_pod)]
    ]
    return _mock_operator

@pytest.fixture
def mock_update():
    return Mock()

@pytest.fixture
def mock_sign_request():
    _mock = Mock()
    _mock.return_value = ({}, None)
    return _mock

@pytest.fixture
def mock_validator_by_hotkey(mock_validator):
    _mock = Mock()
    _mock.return_value = mock_validator
    return _mock

@pytest.fixture
def mock_stop_server_monitoring():
    return AsyncMock()

@pytest.fixture
def mock_start_server_monitoring():
    return AsyncMock()

@pytest.fixture
def mock_multicluster_kubeconfig():
    return Mock()

@pytest.fixture
def mock_track_server(mock_node, mock_server):
    _mock = AsyncMock()
    _mock.return_value = (mock_node, mock_server)
    return _mock

@pytest.fixture
def mock_logger():
    return Mock()

@pytest.fixture
def mock_sse_message():
    _mock = Mock()
    _mock.side_effect = lambda msg: msg
    return _mock

@pytest.fixture
def mock_select():
    return Mock()

@pytest.fixture(autouse=True)
def mock_dependencies(
    mock_update, mock_sign_request, mock_validator_by_hotkey,
    mock_settings, mock_k8s_operator, mock_get_db_session,
    mock_stop_server_monitoring, mock_start_server_monitoring,
    mock_track_server, mock_logger, mock_sse_message,
    mock_multicluster_kubeconfig, mock_select, mock_gpu
):
    """Mock all external dependencies using constants for maintainability"""
    # Create dictionary mapping constants to mock instances

    mock_server = Mock(spec=Server)

    dependency_config = {
        "chutes_miner.api.server.util": {
            MockDependencies.K8S_OPERATOR: mock_k8s_operator,
            MockDependencies.GET_SESSION: mock_get_db_session,
            MockDependencies.SERVER: mock_server,
            MockDependencies.GPU: mock_gpu,
            MockDependencies.UPDATE: mock_update,
            MockDependencies.SIGN_REQUEST: mock_sign_request,
            MockDependencies.STOP_SERVER_MONITORING: mock_stop_server_monitoring,
            MockDependencies.START_SERVER_MONITORING: mock_start_server_monitoring,
            MockDependencies.MULTI_CLUSTER_KUBE_CONFIG: mock_multicluster_kubeconfig,
            MockDependencies.TRACK_SERVER: mock_track_server,
            MockDependencies.SETTINGS: mock_settings,
            MockDependencies.LOGGER: mock_logger,
            MockDependencies.SSE_MESSAGE: mock_sse_message,
        },
        "chutes_miner.api.server.verification": {
            MockStrategyDependencies.K8S_OPERATOR: mock_k8s_operator,
            MockStrategyDependencies.GET_SESSION: mock_get_db_session,
            MockStrategyDependencies.SELECT: mock_select,
            MockStrategyDependencies.UPDATE: mock_update,
            MockStrategyDependencies.VALIDATOR_BY_HOTKEY: mock_validator_by_hotkey,
            MockStrategyDependencies.SIGN_REQUEST: mock_sign_request,
            MockStrategyDependencies.SETTINGS: mock_settings,
            MockStrategyDependencies.LOGGER: mock_logger,
            MockStrategyDependencies.SSE_MESSAGE: mock_sse_message,
        }
    }

    
    # Create and start all patches
    patches: list[_patch] = []
    
    for module, dependencies in dependency_config.items():
        for constant, mock_obj in dependencies.items():
            patcher = patch(f'{module}.{constant}', mock_obj)
            patches.append(patcher)
            patcher.start()
    
    try:
        yield dependency_config

    finally:
        # Stop all patches
        for patcher in patches:
            patcher.stop()

@pytest.fixture(autouse=True)
def mock_check_verification_task_status():

    _mock = AsyncMock()
    with patch("chutes_miner.api.server.verification.GravalVerificationStrategy._check_verification_task_status", _mock):
        # patch("chutes_miner.api.server.verification.TEEVerificationStrategy._check_verification_task_status", _mock):
        _mock.return_value = True
        yield _mock

@pytest.fixture
def mock_graval_service():
    mock_graval_svc = Mock()
    mock_port = Mock()
    mock_port.node_port = 8080
    mock_graval_svc.spec.ports = [mock_port]

    return mock_graval_svc

# @pytest.fixture(autouse=True)
# def mock_check_attestation_service():
#     with patch("chutes_miner.api.server.verification.VerificationStrategy._check_attestation_service") as mock_check:
#         mock_check.return_value = False
#         yield mock_check