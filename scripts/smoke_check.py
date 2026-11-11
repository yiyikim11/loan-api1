"""Poll the live API until this commit is healthy, then score one applicant.

Env:
  API_URL          base URL of the FastAPI service (no trailing slash)
  API_KEY          value for the x-api-key header (optional if the API has none)
  EXPECTED_COMMIT  git SHA that /health must report as git_sha
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

API_URL = os.environ["API_URL"].rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
EXPECTED = os.environ["EXPECTED_COMMIT"]
GOOD = {
    "amount_usd": 1500,
    "term_months": 12,
    "monthly_income_usd": 350.0,
    "previous_loans": 1,
    "late_payments": 0,
}


def call(method: str, path: str, payload: dict | None = None) -> tuple[int, dict | str]:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if API_KEY:
        headers["x-api-key"] = API_KEY
    req = urllib.request.Request(f"{API_URL}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, body


def main() -> int:
    print(f"Waiting for {API_URL}/health to report git_sha={EXPECTED}")
    for attempt in range(1, 31):
        try:
            status, body = call("GET", "/health")
            live = body.get("git_sha") if isinstance(body, dict) else None
            if status == 200 and live == EXPECTED:
                print(f"Healthy on attempt {attempt}: {body}")
                break
            print(f"[{attempt}/30] status={status} git_sha={live!r}")
        except Exception as exc:  # noqa: BLE001 — cold starts / network blips
            print(f"[{attempt}/30] {exc}")
        time.sleep(20)
    else:
        print("Timed out waiting for the expected commit", file=sys.stderr)
        return 1

    status, body = call("POST", "/api/v1/predictions", GOOD)
    if status != 200:
        print(f"prediction failed: {status} {body}", file=sys.stderr)
        return 1
    assert isinstance(body, dict)
    p = body["default_probability"]
    assert 0 <= p <= 1, body
    assert body["risk_band"] in {"low", "medium", "high"}, body
    print(f"Smoke OK: probability={p} band={body['risk_band']} version={body.get('model_version')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
