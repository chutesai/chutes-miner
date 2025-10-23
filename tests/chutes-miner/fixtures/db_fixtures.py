from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest


@pytest.fixture
def mock_db_session():
    # Create a list of paths where k8s_core_client is imported
    import_paths = ["chutes_miner.api.k8s.operator.get_session"]

    # Create a specific __aexit__ function that returns False only when an exception is raised
    async def mock_aexit(self, exc_type, exc_val, exc_tb):
        # Return False only if there's an exception (exc_type is not None)
        # Otherwise return True for normal operation
        return exc_type is None

    session = MagicMock()
    mock_result = MagicMock()
    mock_result.unique.return_value = mock_result
    mock_result.scalar_one_or_none.return_value = []
    session.execute = AsyncMock(return_value=mock_result)
    session.where = AsyncMock(return_value=mock_result)
    session.commit = AsyncMock()
    session.delete = AsyncMock()
    session.refresh = AsyncMock()

    mock_get_session = AsyncMock(__aenter__=AsyncMock(return_value=session), __aexit__=mock_aexit)

    # Create and start patches for each import path, all returning the same mock
    patches = []
    for path in import_paths:
        patcher = patch(path, return_value=mock_get_session)
        patcher.start()
        patches.append(patcher)

    # Yield the shared mock for use in tests
    yield session

    # Stop all patches when done
    for patcher in patches:
        patcher.stop()

# TODO: Update uses of mock_db_session from above to use this instead
@pytest.fixture
def get_mock_db_session():

    # Create a specific __aexit__ function that returns False only when an exception is raised
    async def mock_aexit(self, exc_type, exc_val, exc_tb):
        # Return False only if there's an exception (exc_type is not None)
        # Otherwise return True for normal operation
        return exc_type is None

    session = MagicMock()
    mock_execute_result = MagicMock()
    mock_execute_result.unique.return_value = mock_execute_result
    mock_execute_result.scalars.return_value = mock_execute_result
    mock_execute_result.scalar_one_or_none.return_value = []
    mock_execute_result.all.return_value = []
    session.execute = AsyncMock(return_value=mock_execute_result)
    session.where = AsyncMock(return_value=mock_execute_result)
    session.commit = AsyncMock()
    session.delete = AsyncMock()
    session.refresh = AsyncMock()

    return session

@pytest.fixture
def set_mock_db_session_result(get_mock_db_session):

    def _set_mock_db_session_result(query_data: list[Any] = []):
        get_mock_db_session.execute.return_value.scalar_one_or_none.return_value = query_data[0]
        get_mock_db_session.execute.return_value.all.return_value = query_data

    return _set_mock_db_session_result

@pytest.fixture
def mock_get_db_session(get_mock_db_session):
    # Create a specific __aexit__ function that returns False only when an exception is raised
    async def mock_aexit(self, exc_type, exc_val, exc_tb):
        # Return False only if there's an exception (exc_type is not None)
        # Otherwise return True for normal operation
        return exc_type is None

    session = get_mock_db_session

    mock_context_manager = AsyncMock(__aenter__=AsyncMock(return_value=session), __aexit__=mock_aexit)
    mock_get_session = Mock(return_value=mock_context_manager)

    # Yield the shared mock for use in tests
    return mock_get_session
