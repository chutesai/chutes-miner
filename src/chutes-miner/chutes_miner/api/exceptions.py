class DuplicateServer(Exception): ...


class NonEmptyServer(Exception): ...


class GPUlessServer(Exception): ...


class DeploymentFailure(Exception): ...


class BootstrapFailure(Exception): ...


class GraValBootstrapFailure(BootstrapFailure): ...


class TEEBootstrapFailure(BootstrapFailure): ...


class AgentError(Exception):
    """Exception raised when agent encounters an error (e.g., no active monitoring state - 409 Conflict)"""
    def __init__(self, response_text: str, status_code: int = 409):
        self.status_code = status_code
        self.response_text = response_text
        message = (
            f"Agent error (status_code={status_code}). "
            f"Agent cannot remove itself from cache. response={response_text}"
        )
        super().__init__(message)
