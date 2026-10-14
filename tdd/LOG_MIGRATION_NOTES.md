# Miner CLI log migration

Branch `vm-log-read-cli`.

## What changed

- `src/chutes-miner-cli/chutes_miner_cli/log_read.py` (added). `read_miner_logs` GETs `{validator_api}/logs/miner` with no query string. `authorization_token` sets `Authorization` to `jwt.strip().split(" ")[-1]`, so `Bearer eyJ...` is sent as `eyJ...`. A `cursor` argument is deleted and is not placed on the URL or headers. A 200 body is parsed with the standard-library `json` and passed to `render_log_lines`, which prints each non-empty string `log` in order. 401 returns `Not authorized to read logs.` 403 and 404 return the JSON `detail`. Any other status returns `HTTP {status}: {body}`. A non-list body or invalid JSON returns empty text and no error. `SessionClient` adapts an aiohttp session.
- `src/chutes-miner-cli/chutes_miner_cli/cli.py`. `instance_logs` (`chutes-miner instance-logs`) no longer calls `GET /miner/instance_logs`. Removed `--cursor` / `-c`, `_INSTANCE_LOGS_CURSOR_RE`, `_process_log_line` (SSE `data:` parsing and cursor tracking), and `_emit_log_cursor` (the Ctrl+C and stream-error resume hint). `import re` left with the regex. The command calls `read_miner_logs` once, prints `result.text` to stdout, and exits 1 on `result.error` or a connect/read error. KeyboardInterrupt still exits 130, with no resume hint. The command help now says "Read pre-activation instance logs".
- `pyproject.toml`. Appended `[tool.pytest.ini_options]` with pytest's default `norecursedirs` plus `"tdd"`. Poetry tables are unchanged.
- `tdd/tests/test_log_read_cli.py` (added). Hand-written `HttpSpy`. No `conftest.py`.

## What was not changed

- `--jwt` / `-j` and `CHUTES_LAUNCH_JWT` are still the only credential. The CLI does not re-sign the token and does not check expiry locally.
- `--validator-api` and `VALIDATOR_API` are unchanged. The default base URL is still `https://api.chutes.ai`.
- Every other miner command is unchanged.
- There is no client-side `activated_at` filter. `GET /logs/miner` already drops lines at or after `activated_at`.

## Test coverage

Tests live in `tdd/tests/test_log_read_cli.py`. They build their own spy, import nothing from `tests/`, and `tdd` is in `norecursedirs`, so a bare `pytest` does not collect them.

- `test_should_render_log_text_when_response_has_one_line` — one object renders `booting\n`.
- `test_should_preserve_order_when_response_has_multiple_lines` — two objects render `alpha` then `beta`.
- `test_should_skip_line_when_log_field_is_missing_or_null` — a missing or null `log` is omitted; a null `ts` or `stream` does not drop a present `log`.
- `test_should_render_empty_text_when_response_has_no_lines` — `[]` renders `""`.
- `test_should_report_auth_error_when_response_is_401` — error is `Not authorized to read logs.` and text is empty.
- `test_should_include_server_detail_when_response_is_403` — the error contains `Logs are only available for public chutes.`
- `test_should_include_server_detail_when_response_is_404` — the error contains `Unknown launch config.`
- `test_should_report_http_status_when_response_is_unexpected` — HTTP 500 with body `boom` returns `HTTP 500: boom`.
- `test_should_request_miner_logs_with_bare_token_and_no_query` — a trailing slash on the base URL is stripped; the spy sees `https://api.chutes.ai/logs/miner`, `Authorization: eyJtoken`, and `params is None`.
- `test_should_strip_scheme_prefix_from_authorization_header` — `Bearer eyJabc` is sent as `Authorization: eyJabc`.
- `test_should_omit_cursor_from_the_request` — passing `cursor="2026-04-01T17:53:05"` leaves it off the path, params, and headers.
- `test_should_return_rendered_lines_when_response_is_200` — a 200 array returns `booting\n` and `error is None`.
- `test_should_return_empty_text_when_body_is_not_a_list` — a 200 JSON object returns empty text and no error.
- `test_should_return_empty_text_when_body_is_not_json` — a 200 body of `not-json` returns empty text and no error.

## Depends on

`chutes-api` branch `vm-log-read-routes` must merge first. `GET /logs/miner` must be deployed before this CLI change is released.
