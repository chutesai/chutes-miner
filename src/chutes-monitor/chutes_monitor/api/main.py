# main.py
"""
Control Plane API entrypoint for K3s cluster monitoring.
"""

import hashlib
import asyncio
from contextlib import asynccontextmanager
from chutes_monitor.cluster_monitor import HealthChecker
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse
from loguru import logger

from chutes_monitor.api.cluster.router import router as cluster_router
from chutes_monitor.api.monitoring.router import router as monitoring_router
from chutes_common.redis import MonitoringRedisClient
from chutes_common.constants import CLUSTER_ENDPOINT, MONITORING_ENDPOINT
import chutes_common.schemas.orms  # noqa: F401
import os

# Configure logging
logger.remove()
logger.add(
    sink=lambda message: print(message, end=""),
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="{time} | {level} | {name}:{function}:{line} | {message}",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events"""
    # Startup
    logger.info("Starting Control Plane API server")

    # Initialize Redis
    redis_client = MonitoringRedisClient()

    # Initialize health checker
    health_checker = HealthChecker()
    health_checker.start()

    logger.info("Control Plane API server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Control Plane API server")

    # Close Redis connection
    redis_client.close()
    await health_checker.stop()

    logger.info("Control Plane API server stopped")


app = FastAPI(
    title="Chutes Agent Monitor",
    version="1.0.0",
    description="Central monitoring and management for chutes clusters",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

# Include routers
app.include_router(cluster_router, prefix=CLUSTER_ENDPOINT, tags=["Clusters"])
app.include_router(monitoring_router, prefix=MONITORING_ENDPOINT, tags=["Monitoring"])


@app.get("/health")
async def health_check():
    """Health check endpoint for the control plane"""
    return {"status": "healthy", "service": "agent-monitor"}


@app.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"message": "pong"}


@app.middleware("http")
async def request_body_checksum(request: Request, call_next):
    """Add request body checksum for security/debugging"""
    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
        sha256_hash = hashlib.sha256(body).hexdigest()
        request.state.body_sha256 = sha256_hash
    else:
        request.state.body_sha256 = None
    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests"""
    start_time = asyncio.get_event_loop().time()

    response = await call_next(request)

    process_time = asyncio.get_event_loop().time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s"
    )

    return response
