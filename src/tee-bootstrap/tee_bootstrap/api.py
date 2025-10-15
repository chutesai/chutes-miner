from abc import abstractmethod
from fastapi import FastAPI
from loguru import logger
import uvicorn

class WebServer:
    """Async web server for admission webhook using FastAPI."""

    def __init__(self):
        self.app = FastAPI()
        self._setup_routes()

    def _setup_routes(self):
        """
        Setup web routes.
        Example: 
        self.app.add_api_route('/route', self.handle_route, methods=["GET"])
        """
        self.app.add_api_route("/quote", self.get_tdx_quote, methods=["GET"])

    def run(self):
        """Run the webhook server."""
        # Build kwargs dynamically for uvicorn.run
        uvicorn_kwargs = {}

        if self.config.uds_path:
            logger.info(f"Starting admission webhook server on Unix socket {self.config.uds_path}")
            uvicorn_kwargs["uds"] = self.config.uds_path
        else:
            logger.info(
                f"Starting admission webhook server on {self.config.bind_address}:{self.config.port}"
            )
            uvicorn_kwargs["host"] = self.config.bind_address
            uvicorn_kwargs["port"] = self.config.port
            # Apply TLS if configured for TCP
            if self.config.tls_cert_path and self.config.tls_key_path:
                uvicorn_kwargs["ssl_certfile"] = self.config.tls_cert_path
                uvicorn_kwargs["ssl_keyfile"] = self.config.tls_key_path
                logger.info("TLS enabled")

        uvicorn.run(
            self.app,
            log_level="debug" if self.config.debug else "info",
            **uvicorn_kwargs
        )

    async def get_tdx_quote(self):
        pass