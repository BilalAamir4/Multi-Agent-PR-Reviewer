"""A small GitHub REST client for stage 6 (decision 0016): read a pull request, list its reviews,
create one review with inline comments. The token is sent only in the ``Authorization`` header and
is stripped from every error text (``scrub``)."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx

API_VERSION = "2022-11-28"
RETRY_STATUS = frozenset({500, 502, 503, 504})
MAX_WAIT_S = 120.0


class GitHubError(Exception):
    """An API failure. ``str(e)`` never contains the token."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def scrub(text: object, token: str | None = None) -> str:
    """``text`` with the token, any ``Bearer ...`` string and any GitHub token shape replaced by
    ``***``."""
    out = str(text)
    if token:
        out = out.replace(token, "***")
    out = re.sub(r"(?i)bearer\s+[A-Za-z0-9_\-\.]+", "Bearer ***", out)
    return re.sub(r"\b(github_pat_|gh[pousr]_)[A-Za-z0-9_]+", "***", out)


class GitHubClient:
    def __init__(
        self,
        token: str,
        base_url: str = "https://api.github.com",
        *,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
    ):
        if not token:
            raise GitHubError("GITHUB_TOKEN is empty")
        self._token = token
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=30.0,
            transport=transport,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": API_VERSION,
                "User-Agent": "mapr-approval",
            },
        )
        self._max_retries = max_retries
        self._sleep = sleep
        self._now = now

    def close(self) -> None:
        self._http.close()

    def __repr__(self) -> str:  # never show the token
        return f"GitHubClient(base_url={str(self._http.base_url)!r})"

    # --- transport -------------------------------------------------------------------------

    def _delay(self, resp: httpx.Response, attempt: int) -> float:
        retry_after = resp.headers.get("retry-after", "").strip()
        if retry_after.isdigit():
            return min(float(retry_after), MAX_WAIT_S)
        reset = resp.headers.get("x-ratelimit-reset", "")
        if reset.isdigit() and resp.headers.get("x-ratelimit-remaining") == "0":
            return min(max(float(reset) - self._now(), 1.0), MAX_WAIT_S)
        return min(float(2**attempt), MAX_WAIT_S)

    @staticmethod
    def _rate_limited(resp: httpx.Response) -> bool:
        if resp.status_code == 429:
            return True
        if resp.status_code != 403:
            return False
        if resp.headers.get("retry-after") or resp.headers.get("x-ratelimit-remaining") == "0":
            return True
        return "secondary rate limit" in resp.text.lower()

    def _request(
        self, method: str, path: str, *, retry_5xx: bool = True, **kw: Any
    ) -> httpx.Response:
        """Retries 429 and rate limits (primary and secondary) always, and 5xx and network errors
        only when ``retry_5xx`` (a POST may have been processed, so it is not retried)."""
        last = ""
        for attempt in range(self._max_retries + 1):
            can_retry = attempt < self._max_retries
            try:
                resp = self._http.request(method, path, **kw)
            except httpx.TransportError as e:
                last = f"{type(e).__name__}: {scrub(e, self._token)}"
                if retry_5xx and can_retry:
                    self._sleep(min(float(2**attempt), MAX_WAIT_S))
                    continue
                raise GitHubError(f"{method} {path} failed: {last}") from None
            retryable = self._rate_limited(resp) or (retry_5xx and resp.status_code in RETRY_STATUS)
            if retryable and can_retry:
                self._sleep(self._delay(resp, attempt))
                continue
            if resp.status_code >= 400:
                try:
                    detail = resp.json().get("message", "")
                except (ValueError, AttributeError):
                    detail = resp.text[:200]
                raise GitHubError(
                    scrub(f"{method} {path} -> {resp.status_code}: {detail}", self._token),
                    resp.status_code,
                )
            return resp
        raise GitHubError(f"{method} {path} failed after retries: {last}")  # pragma: no cover

    # --- API -------------------------------------------------------------------------------

    def get_pr(self, owner: str, repo: str, number: int) -> dict[str, Any]:
        data = self._request("GET", f"/repos/{owner}/{repo}/pulls/{number}").json()
        head_repo = data["head"].get("repo") or {}
        if not head_repo.get("clone_url"):
            raise GitHubError("the pull request's head repository is gone (deleted fork?)")
        return {
            "head_sha": data["head"]["sha"],
            "base_sha": data["base"]["sha"],
            "title": data.get("title") or "",
            "body": data.get("body") or "",
            "clone_url": head_repo["clone_url"],
            "html_url": data.get("html_url", ""),
            "state": data.get("state", ""),
        }

    def get_pr_diff(self, owner: str, repo: str, number: int) -> str:
        resp = self._request(
            "GET",
            f"/repos/{owner}/{repo}/pulls/{number}",
            headers={"Accept": "application/vnd.github.diff"},
        )
        return resp.content.decode("utf-8")

    def list_reviews(self, owner: str, repo: str, number: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for page in range(1, 21):
            batch = self._request(
                "GET",
                f"/repos/{owner}/{repo}/pulls/{number}/reviews",
                params={"per_page": 100, "page": page},
            ).json()
            out.extend(batch)
            if len(batch) < 100:
                break
        return out

    def list_review_comments(
        self, owner: str, repo: str, number: int, review_id: int
    ) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            f"/repos/{owner}/{repo}/pulls/{number}/reviews/{review_id}/comments",
            params={"per_page": 100},
        ).json()

    def create_review(
        self,
        owner: str,
        repo: str,
        number: int,
        commit_id: str,
        body: str,
        comments: Sequence[dict[str, Any]],
        event: str = "COMMENT",
    ) -> dict[str, Any]:
        payload = {"commit_id": commit_id, "body": body, "event": event, "comments": list(comments)}
        return self._request(
            "POST", f"/repos/{owner}/{repo}/pulls/{number}/reviews", json=payload, retry_5xx=False
        ).json()
