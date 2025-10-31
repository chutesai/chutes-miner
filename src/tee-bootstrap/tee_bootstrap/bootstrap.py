import aiohttp
import asyncio
import base64
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def get_tdx_quote(nonce: str, socket_path: str = "/var/run/tdx-quote.sock") -> bytes:
    """
    Send a GET request to the TdxQuoteServer over a Unix socket using aiohttp.

    Args:
        nonce: The nonce to include in the quote request.
        socket_path: Path to the Unix socket (default: /var/run/tdx-quote.sock).

    Returns:
        bytes: The decoded TDX quote.

    Raises:
        aiohttp.ClientConnectorError: If the connection to the socket fails.
        aiohttp.ClientResponseError: If the server returns a 4xx or 5xx status code.
        ValueError: If the response is not valid base64.
        FileNotFoundError: If the socket file does not exist.
    """
    # Verify socket path exists
    if not Path(socket_path).exists():
        raise FileNotFoundError(f"Unix socket not found: {socket_path}")

    try:
        # Create a connector for the Unix socket
        connector = aiohttp.UnixConnector(path=socket_path)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Send GET request to /quote with nonce query parameter
            async with session.get("http://localhost/quote", params={"nonce": nonce}) as response:
                response.raise_for_status()  # Raises an exception for 4xx/5xx responses
                # Read the base64-encoded response
                response_text = await response.text()
                # Decode the base64 response
                try:
                    quote = base64.b64decode(response_text)
                    logger.info(
                        f"Successfully retrieved quote for nonce '{nonce}' (length: {len(quote)} bytes)"
                    )
                    return quote
                except base64.binascii.Error as e:
                    logger.error(f"Failed to decode base64 response: {e}")
                    raise ValueError("Invalid base64 response from server")
    except aiohttp.ClientConnectorError as e:
        logger.error(f"Failed to connect to socket {socket_path}: {e}")
        raise
    except aiohttp.ClientResponseError as e:
        logger.error(f"HTTP error: {e.status} - {e.message}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


async def main():
    """Main function to demonstrate the quote request."""
    nonce = "test"
    socket_path = "/var/run/tdx-quote.sock"
    try:
        quote = await get_tdx_quote(nonce, socket_path)
        # Optionally save the quote to a file
        with open("quote.bin", "wb") as f:
            f.write(quote)
        logger.info("Quote saved to quote.bin")
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
