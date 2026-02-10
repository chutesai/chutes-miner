"""
Utility/helper functions.
"""

import datetime
import re
import orjson as json
import semver


def now_str():
    """
    Return current (UTC) timestamp as string.
    """
    return datetime.datetime.utcnow().isoformat()


def sse(data):
    """
    Format response object for server-side events stream.
    """
    return f"data: {json.dumps(data).decode()}\n\n"


def sse_message(message):
    """
    Format a simple trace message with timestamp.
    """
    return sse(
        {
            "timestamp": now_str(),
            "message": message,
        }
    )

def semcomp(input_version: str, target_version: str):
    """
    Semver comparison with cleanup.
    """
    if not input_version:
        input_version = "0.0.0"
    re_match = re.match(r"^([0-9]+\.[0-9]+\.[0-9]+).*", input_version)
    clean_version = re_match.group(1) if re_match else "0.0.0"
    return semver.compare(clean_version, target_version)