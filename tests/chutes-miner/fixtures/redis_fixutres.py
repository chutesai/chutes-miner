from unittest.mock import MagicMock, Mock, patch

import pytest
from redis import Redis

from chutes_common.k8s import ClusterResources
from chutes_common.redis import MonitoringRedisClient
from chutes_common.settings import RedisSettings

settings = RedisSettings()

@pytest.fixture(autouse=True)
def mock_redis():
    """Mock Redis client"""
    with patch('chutes_common.redis.Redis') as mock_redis_class:
        mock_redis_instance = Mock(spec=Redis)
        mock_redis_class.from_url.return_value = mock_redis_instance
        yield mock_redis_instance

@pytest.fixture()
def mock_redis_client(mock_redis):
    """Create MonitoringRedisClient instance with mocked dependencies"""
    # Reset singleton instance
    with patch('chutes_miner.api.k8s.operator.MonitoringRedisClient') as mock_redis_class:
        mock_redis_client = MagicMock()

        # Provide sensible defaults for lookups so code paths can execute without extra setup
        mock_redis_client.get_resource_with_context.return_value = ("test-cluster", None)
        mock_redis_client.get_resources.return_value = ClusterResources()
        mock_redis_client.get_all_cluster_names.return_value = ["test-cluster"]

        mock_redis_class.return_value = mock_redis_client
        yield mock_redis_client

@pytest.fixture
def redis_client(mock_redis):
    """Create MonitoringRedisClient instance with mocked dependencies"""
    # Reset singleton instance
    MonitoringRedisClient._instance = None
    client = MonitoringRedisClient(settings)
    return client