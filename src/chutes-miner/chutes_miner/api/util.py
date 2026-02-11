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

def semcomp(input_version: str, target_version: str) -> int:
    """
    Semver comparison. Strips prerelease/rc suffix by using only the X.Y.Z prefix.

    Returns: -1 if input_version < target_version, 0 if equal, 1 if greater.
    """
    if not input_version:
        input_version = "0.0.0"
    # Normalize ".rc" to "-rc" so we can strip cleanly (e.g. "0.6.0.rc0" -> "0.6.0-rc0")
    if ".rc" in input_version and "-rc" not in input_version:
        prefix, suffix = input_version.split(".rc", 1)
        if prefix and suffix:
            input_version = f"{prefix}-rc{suffix}"
    # Use only the X.Y.Z prefix for comparison (strips -rc0, etc.)
    re_match = re.match(r"^([0-9]+\.[0-9]+\.[0-9]+)", input_version)
    clean_version = re_match.group(1) if re_match else "0.0.0"
    return semver.compare(clean_version, target_version)