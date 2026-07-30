import pytest

from fixtures.bootstrap_fixtures import *  # noqa

# Import the real exception AFTER the star import above: bootstrap_fixtures defines its own
# stub GraValBootstrapFailure, and `import *` would otherwise shadow the production class,
# breaking the pytest.raises match below.
from chutes_miner.api.exceptions import GraValBootstrapFailure  # noqa: E402
from chutes_miner.api.server.verification import (  # noqa: E402
    TEEVerificationStrategy,
    VerificationStrategy,
)


@pytest.mark.asyncio
async def test_create_blocks_non_tee_node(mock_node, mock_server_args, mock_server):
    """A node without chutes/tee=true must be rejected (network is TEE-exclusive)."""
    mock_node.metadata.labels.pop("chutes/tee", None)
    with pytest.raises(GraValBootstrapFailure):
        await VerificationStrategy.create(mock_node, mock_server_args, mock_server)


@pytest.mark.asyncio
async def test_create_returns_tee_strategy_for_tee_node(
    mock_node, mock_server_args, mock_server
):
    """A node labeled chutes/tee=true gets the TEE verification strategy."""
    mock_node.metadata.labels["chutes/tee"] = "true"
    strategy = await VerificationStrategy.create(mock_node, mock_server_args, mock_server)
    assert isinstance(strategy, TEEVerificationStrategy)
