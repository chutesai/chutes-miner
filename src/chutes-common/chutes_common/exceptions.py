class ClusterNotFoundException(Exception): ...


class ClusterConflictException(Exception): ...


class ClusterRegistrationException(Exception): ...


class ServerNotFoundException(Exception): ...


class AgentError(Exception):
    """Raised when an agent API request fails (e.g. /monitor/start or /monitor/stop)."""

    def __init__(self, response_text: str, status_code: int = 409):
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(
            f"Agent error (status_code={status_code}). response={response_text}"
        )
