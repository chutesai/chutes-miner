import json


def log_read_failure(status, body):
    if status == 401:
        return "Not authorized to read logs."
    if status == 403 or status == 404:
        return json.loads(body)["detail"]
    return f"HTTP {status}: {body}"


class LogRead:
    def __init__(self, text, error):
        self.text = text
        self.error = error


class SessionClient:
    """Adapt an aiohttp session to the client `read_miner_logs` calls."""

    def __init__(self, session):
        self._session = session

    async def get(self, path, headers=None, params=None):
        kwargs = {"headers": headers}
        if params is not None:
            kwargs["params"] = params
        async with self._session.get(path, **kwargs) as response:
            body = await response.text()
            return _BodyResponse(response.status, body)


class _BodyResponse:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    async def text(self):
        return self._body


def miner_logs_url(validator_api):
    return f"{validator_api.rstrip('/')}/logs/miner"


def authorization_token(jwt):
    return jwt.strip().split(" ")[-1]


async def read_miner_logs(client, validator_api, jwt, cursor=None):
    del cursor
    response = await client.get(
        miner_logs_url(validator_api),
        headers={"Authorization": authorization_token(jwt)},
    )
    body = await response.text()
    if response.status == 200:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return LogRead("", None)
        return LogRead(render_log_lines(payload), None)
    return LogRead("", log_read_failure(response.status, body))


def render_log_lines(lines):
    if not isinstance(lines, list):
        return ""
    rendered = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        log = line.get("log")
        if isinstance(log, str) and log:
            rendered.append(f"{log}\n")
    return "".join(rendered)
