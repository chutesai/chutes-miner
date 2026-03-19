"""
Server bootstrap orchestration.

Shared verification flow. Caller handles tracking (miner API: track_server +
monitoring; registration API: validator inventory). Bootstrap only runs
verification - caller must provide server context and handles cleanup on failure.
"""

import asyncio
import time
import traceback
from typing import AsyncGenerator, Optional

from chutes_miner.api.k8s.config import KubeConfig, MultiClusterKubeConfig
from chutes_miner.api.server.util import (
    start_server_monitoring,
    stop_server_monitoring,
    track_server,
)
from chutes_miner.common.verification import VerificationStrategy
from chutes_miner.api.util import sse_message
from chutes_common.schemas.server import Server, ServerArgs
from chutes_miner.api.database import get_session
from chutes_miner.api.exceptions import VerificationFailure
from kubernetes.client import V1Node
from loguru import logger
from sqlalchemy import update

from chutes_common.schemas.gpu import GPU


async def verify_server(
    node_object: V1Node,
    server_args: ServerArgs,
    kubeconfig: Optional[KubeConfig],
    server: Server,
) -> AsyncGenerator[str, None]:
    """
    Verify server (GraVal/TEE). Caller provides server and handles tracking.

    Orchestrates:
    1. Add kubeconfig to multi-cluster config (if remote cluster)
    2. Run verification (GraVal or TEE)
    3. Mark GPUs as verified on success

    Caller is responsible for:
    - Providing server (from track_server for miner API, or equivalent for registration API)
    - Tracking on success (miner: already done before call; registration: advertise)
    - Cleanup on failure is handled here (strategy.cleanup purges validator, deletes server)

    Args:
        node_object: Kubernetes V1Node for the GPU node
        server_args: Server configuration (validator, name, gpu_short_ref, etc.)
        kubeconfig: Kubeconfig for remote cluster, or None for in-cluster
        server: Server context (caller provides - from track_server or equivalent)

    Yields:
        SSE-formatted status messages
    """
    started_at = time.time()
    strategy = None
    success = False

    async def _cleanup(delete_node: bool = False):
        if delete_node and server_args.agent_api:
            await stop_server_monitoring(server_args.agent_api)
            logger.info(f"Stopped monitoring for {server_args.name}")

    yield sse_message(
        f"provisioning verification resources for server_id={node_object.metadata.uid}...",
    )

    try:
        if kubeconfig:
            MultiClusterKubeConfig().add_config(kubeconfig)

        strategy = await VerificationStrategy.create(node_object, server_args, server)

        task = asyncio.create_task(strategy.run())

        async for msg in strategy.stream_messages():
            yield msg

        await task

        async with get_session() as session:
            await session.execute(
                update(GPU)
                .where(GPU.server_id == node_object.metadata.uid)
                .values({"verified": True})
            )
            await session.commit()

        yield sse_message(f"completed server bootstrapping in {time.time() - started_at} seconds!")
        success = True

    except VerificationFailure:
        pass
    except Exception:
        error_message = f"Unhandled exception bootstrapping new node:\n{traceback.format_exc()}"
        logger.error(error_message)
        yield sse_message(error_message)
        raise
    finally:
        delete_node = not success
        if strategy:
            await strategy.cleanup(delete_node=delete_node)
        await _cleanup(delete_node=delete_node)


async def bootstrap_server(
    node_object: V1Node, server_args: ServerArgs, kubeconfig: Optional[KubeConfig]
) -> AsyncGenerator[str, None]:
    """
    Full miner API add-node flow: track, monitor, then verify.

    Caller handles tracking by doing track_server + start_server_monitoring
    before verification. If verification fails, cleanup purges from validator
    and deletes server from DB.
    """
    yield sse_message(
        f"attempting to add node server_id={node_object.metadata.uid} to inventory...",
    )

    # Miner API: track first (verification needs Server in DB for GPU records)
    node, server = await track_server(
        server_args.validator,
        server_args.hourly_cost,
        node_object,
        add_labels={
            "gpu-short-ref": server_args.gpu_short_ref,
            "chutes/validator": server_args.validator,
            "chutes/worker": "true",
        },
        agent_api=server_args.agent_api,
        kubeconfig=kubeconfig,
    )

    yield sse_message(
        f"server with server_id={node_object.metadata.uid} now tracked in "
        "database, provisioning verification resources...",
    )

    if server_args.agent_api:
        await start_server_monitoring(server_args.agent_api)
        yield sse_message(
            f"Started monitoring server_id={node_object.metadata.uid}...",
        )

    async for msg in verify_server(node, server_args, kubeconfig, server):
        yield msg
