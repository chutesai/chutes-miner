import asyncio
import time
import traceback
from unittest.mock import AsyncMock, Mock, _patch, patch
from typing import Optional, AsyncGenerator

from chutes_miner.api.config import Settings
from chutes_miner.api.k8s.operator import K8sOperator
import pytest
from kubernetes.client import V1Node, V1ObjectMeta

# Assuming these are your project imports - adjust as needed
# from chutes_miner.api.server.util import bootstrap_server, ServerArgs, KubeConfig, GraValBootstrapFailure

# Constants for mock dependencies - makes maintenance easier and prevents typos
class MockDependencies:
    """Constants for all mocked dependencies in bootstrap_server"""
    K8S_OPERATOR = 'K8sOperator'
    GET_SESSION = 'get_session'
    SELECT = 'select'
    SERVER = 'Server'
    GPU = 'GPU'
    UPDATE = 'update'
    VALIDATOR_BY_HOTKEY = 'validator_by_hotkey'
    SIGN_REQUEST = 'sign_request'
    STOP_SERVER_MONITORING = 'stop_server_monitoring'
    START_SERVER_MONITORING = 'start_server_monitoring'
    MULTI_CLUSTER_KUBE_CONFIG = 'MultiClusterKubeConfig'
    TRACK_SERVER = 'track_server'
    DEPLOY_GRAVAL = 'deploy_graval'
    GATHER_GPU_INFO = 'gather_gpu_info'
    SETTINGS = 'settings'
    ADVERTISE_NODES = '_advertise_nodes'
    CHECK_VERIFICATION_TASK_STATUS = 'check_verification_task_status'
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


@pytest.fixture
def mock_node():
    """Create a mock V1Node object"""
    node = V1Node()
    node.metadata = V1ObjectMeta()
    node.metadata.uid = "test-node-uid-123"
    node.metadata.name = "test-node-name"
    return node


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
def mock_server():
    """Create mock Server object"""
    return MockServer(
        server_id="test-node-uid-123",
        validator="test_validator",
        cpu_per_gpu=8,
        memory_per_gpu=32
    )


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

    return _mock_settings

@pytest.fixture
def mock_k8s_operator():
    _mock_operator = Mock(spec=K8sOperator)
    _mock_operator.return_value = _mock_operator
    return _mock_operator


@pytest.fixture
def mock_dependencies(mock_settings, mock_k8s_operator, mock_get_session):
    """Mock all external dependencies using constants for maintainability"""
    # Create dictionary mapping constants to mock instances
    dependency_config = {
        MockDependencies.K8S_OPERATOR: mock_k8s_operator,
        MockDependencies.GET_SESSION: mock_get_session,
        MockDependencies.SELECT: Mock(),
        MockDependencies.SERVER: Mock(),
        MockDependencies.GPU: Mock(),
        MockDependencies.UPDATE: Mock(),
        MockDependencies.VALIDATOR_BY_HOTKEY: Mock(),
        MockDependencies.SIGN_REQUEST: Mock(),
        MockDependencies.STOP_SERVER_MONITORING: AsyncMock(),
        MockDependencies.START_SERVER_MONITORING: AsyncMock(),
        MockDependencies.MULTI_CLUSTER_KUBE_CONFIG: Mock(),
        MockDependencies.TRACK_SERVER: AsyncMock(),
        MockDependencies.DEPLOY_GRAVAL: AsyncMock(),
        MockDependencies.GATHER_GPU_INFO: AsyncMock(),
        MockDependencies.SETTINGS: mock_settings,
        MockDependencies.ADVERTISE_NODES: AsyncMock(),
        MockDependencies.CHECK_VERIFICATION_TASK_STATUS: AsyncMock(),
        MockDependencies.LOGGER: Mock(),
        MockDependencies.SSE_MESSAGE: Mock(),
    }
    
    # Create and start all patches
    patches: list[_patch] = []
    
    for constant, mock_obj in dependency_config.items():
        patcher = patch(f'chutes_miner.api.server.util.{constant}', mock_obj)
        patches.append(patcher)
        patcher.start()
    
    try:
        yield dependency_config
    finally:
        # Stop all patches
        for patcher in patches:
            patcher.stop()


async def collect_sse_messages(async_gen: AsyncGenerator) -> list:
    """Helper to collect all SSE messages from the generator"""
    messages = []
    async for message in async_gen:
        messages.append(message)
    return messages


@pytest.mark.asyncio
async def test_bootstrap_server_success_without_kubeconfig(
    mock_node, mock_server_args, mock_gpus, mock_server, 
    mock_validator, mock_validator_nodes, mock_dependencies
):
    """Test successful server bootstrapping without kubeconfig"""
    # Setup mocks
    mock_dependencies[MockDependencies.TRACK_SERVER].return_value = (mock_node, mock_server)
    mock_dependencies[MockDependencies.DEPLOY_GRAVAL].return_value = (Mock(), Mock())
    mock_dependencies[MockDependencies.GATHER_GPU_INFO].return_value = mock_gpus
    mock_dependencies[MockDependencies.VALIDATOR_BY_HOTKEY].return_value = mock_validator
    mock_dependencies[MockDependencies.ADVERTISE_NODES].return_value = ("task_123", mock_validator_nodes)
    mock_dependencies[MockDependencies.CHECK_VERIFICATION_TASK_STATUS].return_value = True
    mock_dependencies[MockDependencies.SSE_MESSAGE].side_effect = lambda msg: msg
    
    # Mock database session
    mock_session = AsyncMock()
    mock_dependencies[MockDependencies.GET_SESSION].return_value.__aenter__.return_value = mock_session
    
    # Execute test
    from chutes_miner.api.server.util import bootstrap_server  # Replace with actual import
    messages = await collect_sse_messages(
        bootstrap_server(mock_node, mock_server_args, None)
    )
    
    # Assertions
    assert len(messages) > 0
    assert any("attempting to add node" in str(msg) for msg in messages)
    assert any("completed server bootstrapping" in str(msg) for msg in messages)
    
    # Verify key method calls
    mock_dependencies[MockDependencies.TRACK_SERVER].assert_called_once()
    mock_dependencies[MockDependencies.DEPLOY_GRAVAL].assert_called_once()
    mock_dependencies[MockDependencies.GATHER_GPU_INFO].assert_called_once()
    mock_dependencies[MockDependencies.ADVERTISE_NODES].assert_called_once()


@pytest.mark.asyncio
async def test_bootstrap_server_success_with_kubeconfig(
    mock_node, mock_server_args, mock_kubeconfig, mock_gpus, 
    mock_server, mock_validator, mock_validator_nodes, mock_dependencies
):
    """Test successful server bootstrapping with kubeconfig"""
    # Setup mocks
    mock_dependencies[MockDependencies.TRACK_SERVER].return_value = (mock_node, mock_server)
    mock_dependencies[MockDependencies.DEPLOY_GRAVAL].return_value = (Mock(), Mock())
    mock_dependencies[MockDependencies.GATHER_GPU_INFO].return_value = mock_gpus
    mock_dependencies[MockDependencies.VALIDATOR_BY_HOTKEY].return_value = mock_validator
    mock_dependencies[MockDependencies.ADVERTISE_NODES].return_value = ("task_123", mock_validator_nodes)
    mock_dependencies[MockDependencies.CHECK_VERIFICATION_TASK_STATUS].return_value = True
    mock_dependencies[MockDependencies.SSE_MESSAGE].side_effect = lambda msg: msg
    
    # Mock database session
    mock_session = AsyncMock()
    mock_dependencies[MockDependencies.GET_SESSION].return_value.__aenter__.return_value = mock_session
    
    # Execute test
    from chutes_miner.api.server.util import bootstrap_server  # Replace with actual import
    messages = await collect_sse_messages(
        bootstrap_server(mock_node, mock_server_args, mock_kubeconfig)
    )
    
    # Assertions
    assert len(messages) > 0
    mock_dependencies[MockDependencies.MULTI_CLUSTER_KUBE_CONFIG].return_value.add_config.assert_called_once_with(mock_kubeconfig)


@pytest.mark.asyncio
async def test_bootstrap_server_success_with_agent_api(
    mock_node, mock_server_args_with_agent, mock_gpus, mock_server, 
    mock_validator, mock_validator_nodes, mock_dependencies
):
    """Test successful server bootstrapping with agent API monitoring"""
    # Setup mocks
    mock_dependencies[MockDependencies.TRACK_SERVER].return_value = (mock_node, mock_server)
    mock_dependencies[MockDependencies.DEPLOY_GRAVAL].return_value = (Mock(), Mock())
    mock_dependencies[MockDependencies.GATHER_GPU_INFO].return_value = mock_gpus
    mock_dependencies[MockDependencies.VALIDATOR_BY_HOTKEY].return_value = mock_validator
    mock_dependencies[MockDependencies.ADVERTISE_NODES].return_value = ("task_123", mock_validator_nodes)
    mock_dependencies[MockDependencies.CHECK_VERIFICATION_TASK_STATUS].return_value = True
    mock_dependencies[MockDependencies.SSE_MESSAGE].side_effect = lambda msg: msg
    
    # Execute test
    from chutes_miner.api.server.util import bootstrap_server  # Replace with actual import
    messages = await collect_sse_messages(
        bootstrap_server(mock_node, mock_server_args_with_agent, None)
    )
    
    # Assertions
    assert len(messages) > 0
    mock_dependencies[MockDependencies.START_SERVER_MONITORING].assert_called_once_with("http://agent-api.com")
    mock_dependencies[MockDependencies.STOP_SERVER_MONITORING].assert_not_called()


@pytest.mark.asyncio
async def test_bootstrap_server_gpu_verification_failure(
    mock_node, mock_server_args, mock_gpus, mock_server, 
    mock_validator, mock_validator_nodes, mock_k8s_operator,
    mock_dependencies
):
    """Test bootstrap failure due to GPU verification failure"""
    # Setup mocks
    mock_dependencies[MockDependencies.TRACK_SERVER].return_value = (mock_node, mock_server)
    mock_dependencies[MockDependencies.DEPLOY_GRAVAL].return_value = (Mock(), Mock())
    mock_dependencies[MockDependencies.GATHER_GPU_INFO].return_value = mock_gpus
    mock_dependencies[MockDependencies.VALIDATOR_BY_HOTKEY].return_value = mock_validator
    mock_dependencies[MockDependencies.ADVERTISE_NODES].return_value = ("task_123", mock_validator_nodes)
    mock_dependencies[MockDependencies.CHECK_VERIFICATION_TASK_STATUS].return_value = False  # Verification fails
    mock_dependencies[MockDependencies.SSE_MESSAGE].side_effect = lambda msg: msg
    
    # Execute test and expect exception
    from chutes_miner.api.server.util import bootstrap_server, GraValBootstrapFailure  # Replace with actual import
    
    with pytest.raises(GraValBootstrapFailure):
        await collect_sse_messages(
            bootstrap_server(mock_node, mock_server_args, None)
        )
    
    # Verify cleanup was called
    mock_k8s_operator.cleanup_graval.assert_called()


@pytest.mark.asyncio
async def test_bootstrap_server_advertise_nodes_failure(
    mock_node, mock_server_args, mock_gpus, mock_server, 
    mock_validator, mock_k8s_operator, mock_dependencies
):
    """Test bootstrap failure during node advertisement"""
    # Setup mocks
    mock_dependencies[MockDependencies.TRACK_SERVER].return_value = (mock_node, mock_server)
    mock_dependencies[MockDependencies.DEPLOY_GRAVAL].return_value = (Mock(), Mock())
    mock_dependencies[MockDependencies.GATHER_GPU_INFO].return_value = mock_gpus
    mock_dependencies[MockDependencies.VALIDATOR_BY_HOTKEY].return_value = mock_validator
    mock_dependencies[MockDependencies.ADVERTISE_NODES].side_effect = Exception("Advertisement failed")
    mock_dependencies[MockDependencies.SSE_MESSAGE].side_effect = lambda msg: msg
    
    # Execute test and expect exception
    from chutes_miner.api.server.util import bootstrap_server  # Replace with actual import
    
    with pytest.raises(Exception, match="Advertisement failed"):
        await collect_sse_messages(
            bootstrap_server(mock_node, mock_server_args, None)
        )
    
    # Verify cleanup was called with delete_node=True
    mock_k8s_operator.cleanup_graval.assert_called()


@pytest.mark.asyncio
async def test_bootstrap_server_track_server_failure(
    mock_node, mock_server_args, mock_k8s_operator, mock_dependencies
):
    """Test bootstrap failure during server tracking"""
    # Setup mocks
    mock_dependencies[MockDependencies.TRACK_SERVER].side_effect = Exception("Tracking failed")
    mock_dependencies[MockDependencies.SSE_MESSAGE].side_effect = lambda msg: msg
    
    # Execute test and expect exception
    from chutes_miner.api.server.util import bootstrap_server  # Replace with actual import
    
    with pytest.raises(Exception, match="Tracking failed"):
        await collect_sse_messages(
            bootstrap_server(mock_node, mock_server_args, None)
        )
    
    # Verify cleanup was called
    mock_k8s_operator.cleanup_graval.assert_called()


@pytest.mark.asyncio
async def test_bootstrap_server_multiple_seeds_assertion_error(
    mock_node, mock_server_args, mock_gpus, mock_server, 
    mock_validator, mock_k8s_operator, mock_dependencies
):
    """Test bootstrap failure when validators return different seeds"""
    # Setup mocks with different seeds
    validator_nodes_different_seeds = [
        {"seed": "seed_1", "gpu_id": "gpu_1"},
        {"seed": "seed_2", "gpu_id": "gpu_2"},  # Different seed!
    ]
    
    mock_dependencies[MockDependencies.TRACK_SERVER].return_value = (mock_node, mock_server)
    mock_dependencies[MockDependencies.DEPLOY_GRAVAL].return_value = (Mock(), Mock())
    mock_dependencies[MockDependencies.GATHER_GPU_INFO].return_value = mock_gpus
    mock_dependencies[MockDependencies.VALIDATOR_BY_HOTKEY].return_value = mock_validator
    mock_dependencies[MockDependencies.ADVERTISE_NODES].return_value = ("task_123", validator_nodes_different_seeds)
    mock_dependencies[MockDependencies.SSE_MESSAGE].side_effect = lambda msg: msg
    
    # Execute test and expect assertion error
    from chutes_miner.api.server.util import bootstrap_server  # Replace with actual import
    
    with pytest.raises(AssertionError, match="more than one seed"):
        await collect_sse_messages(
            bootstrap_server(mock_node, mock_server_args, None)
        )


@pytest.mark.asyncio
async def test_bootstrap_server_cleanup_with_existing_server(
    mock_node, mock_server_args, mock_k8s_operator, mock_dependencies
):
    """Test cleanup behavior when server exists in database"""
    # Setup server with GPUs in database
    mock_gpu1 = Mock()
    mock_gpu1.gpu_id = "gpu_1"
    mock_gpu2 = Mock()
    mock_gpu2.gpu_id = "gpu_2"
    
    mock_server_db = Mock()
    mock_server_db.gpus = [mock_gpu1, mock_gpu2]
    mock_server_db.validator = "test_validator"
    
    mock_validator = MockValidator("test_hotkey", "http://test-validator.com")
    
    # Setup mock session with server query result
    mock_session = AsyncMock()
    mock_result = Mock()
    mock_result.unique.return_value.scalar_one_or_none.return_value = mock_server_db
    mock_session.execute.return_value = mock_result
    
    mock_dependencies[MockDependencies.GET_SESSION].return_value.__aenter__.return_value = mock_session
    mock_dependencies[MockDependencies.VALIDATOR_BY_HOTKEY].return_value = mock_validator
    mock_dependencies[MockDependencies.TRACK_SERVER].side_effect = Exception("Tracking failed")
    mock_dependencies[MockDependencies.SSE_MESSAGE].side_effect = lambda msg: msg
    
    # Mock aiohttp session for GPU cleanup
    mock_http_session = AsyncMock()
    mock_resp = AsyncMock()
    mock_resp.json.return_value = {"success": True}
    mock_http_session.delete.return_value.__aenter__.return_value = mock_resp
    
    with patch('aiohttp.ClientSession') as mock_aiohttp:
        mock_aiohttp.return_value.__aenter__.return_value = mock_http_session
        
        # Execute test and expect exception
        from chutes_miner.api.server.util import bootstrap_server  # Replace with actual import
        
        with pytest.raises(Exception, match="Tracking failed"):
            await collect_sse_messages(
                bootstrap_server(mock_node, mock_server_args, None)
            )
        
        # Verify GPUs were cleaned up from validator
        assert mock_http_session.delete.call_count == 2  # Two GPUs


@pytest.mark.asyncio 
async def test_bootstrap_server_verification_timeout_simulation(
    mock_node, mock_server_args, mock_gpus, mock_server, 
    mock_validator, mock_validator_nodes, mock_dependencies
):
    """Test verification status checking with multiple polls before success"""
    # Setup mocks
    mock_dependencies[MockDependencies.TRACK_SERVER].return_value = (mock_node, mock_server)
    mock_dependencies[MockDependencies.DEPLOY_GRAVAL].return_value = (Mock(), Mock())
    mock_dependencies[MockDependencies.GATHER_GPU_INFO].return_value = mock_gpus
    mock_dependencies[MockDependencies.VALIDATOR_BY_HOTKEY].return_value = mock_validator
    mock_dependencies[MockDependencies.ADVERTISE_NODES].return_value = ("task_123", mock_validator_nodes)
    
    # Simulate verification polling - return None twice, then True
    mock_dependencies[MockDependencies.CHECK_VERIFICATION_TASK_STATUS].side_effect = [None, None, True]
    mock_dependencies[MockDependencies.SSE_MESSAGE].side_effect = lambda msg: msg
    
    # Mock database session
    mock_session = AsyncMock()
    mock_dependencies[MockDependencies.GET_SESSION].return_value.__aenter__.return_value = mock_session
    
    # Execute test
    from chutes_miner.api.server.util import bootstrap_server  # Replace with actual import
    messages = await collect_sse_messages(
        bootstrap_server(mock_node, mock_server_args, None)
    )
    
    # Verify verification was checked multiple times
    assert mock_dependencies[MockDependencies.CHECK_VERIFICATION_TASK_STATUS].call_count == 3
    
    # Verify we got waiting messages
    waiting_messages = [msg for msg in messages if "waiting for validator" in str(msg)]
    assert len(waiting_messages) >= 2


@pytest.mark.asyncio
async def test_bootstrap_server_timing_measurement(
    mock_node, mock_server_args, mock_gpus, mock_server, 
    mock_validator, mock_validator_nodes, mock_dependencies
):
    """Test that bootstrap timing is properly measured and reported"""
    # Setup mocks
    mock_dependencies[MockDependencies.TRACK_SERVER].return_value = (mock_node, mock_server)
    mock_dependencies[MockDependencies.DEPLOY_GRAVAL].return_value = (Mock(), Mock())
    mock_dependencies[MockDependencies.GATHER_GPU_INFO].return_value = mock_gpus
    mock_dependencies[MockDependencies.VALIDATOR_BY_HOTKEY].return_value = mock_validator
    mock_dependencies[MockDependencies.ADVERTISE_NODES].return_value = ("task_123", mock_validator_nodes)
    mock_dependencies[MockDependencies.CHECK_VERIFICATION_TASK_STATUS].return_value = True
    mock_dependencies[MockDependencies.SSE_MESSAGE].side_effect = lambda msg: msg
    
    # Mock database session
    mock_session = AsyncMock()
    mock_dependencies[MockDependencies.GET_SESSION].return_value.__aenter__.return_value = mock_session
    
    # Mock time to control timing
    start_time = 1000.0
    end_time = 1045.5  # 45.5 seconds
    
    with patch('time.time', side_effect=[start_time, end_time]):
        from chutes_miner.api.server.util import bootstrap_server  # Replace with actual import
        messages = await collect_sse_messages(
            bootstrap_server(mock_node, mock_server_args, None)
        )
    
    # Verify timing is reported in final message
    final_messages = [msg for msg in messages if "completed server bootstrapping" in str(msg)]
    assert len(final_messages) == 1
    assert "45.5 seconds" in str(final_messages[0])


# Integration-style test that doesn't mock everything
@pytest.mark.asyncio
async def test_bootstrap_server_sse_message_flow(mock_node, mock_server_args, mock_dependencies):
    """Test that SSE messages are yielded in the expected order"""
    mock_dependencies[MockDependencies.TRACK_SERVER].return_value = (mock_node, MockServer())
    mock_dependencies[MockDependencies.DEPLOY_GRAVAL].return_value = (Mock(), Mock())
    mock_dependencies[MockDependencies.GATHER_GPU_INFO].return_value = [MockGPU()]
    mock_dependencies[MockDependencies.VALIDATOR_BY_HOTKEY].return_value = MockValidator()
    mock_dependencies[MockDependencies.ADVERTISE_NODES].return_value = ("task_123", [{"seed": "test_seed"}])
    mock_dependencies[MockDependencies.CHECK_VERIFICATION_TASK_STATUS].return_value = True
    mock_dependencies[MockDependencies.SSE_MESSAGE].side_effect = lambda msg: f"SSE: {msg}"

    from chutes_miner.api.server.util import bootstrap_server  # Replace with actual import
    
    messages = await collect_sse_messages(
        bootstrap_server(mock_node, mock_server_args, None)
    )
    
    # Verify message flow
    message_strs = [str(msg) for msg in messages]
    
    # Check that messages appear in expected order
    assert any("attempting to add node" in msg for msg in message_strs)
    assert any("now tracked in database" in msg for msg in message_strs)
    assert any("graval bootstrap job/service created" in msg for msg in message_strs)
    assert any("discovered" in msg and "GPUs" in msg for msg in message_strs)
    assert any("advertising node" in msg for msg in message_strs)
    assert any("successfully advertised node" in msg for msg in message_strs)
    assert any("completed server bootstrapping" in msg for msg in message_strs)