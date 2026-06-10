"""Reddit search fetcher for ticker-specific discussion posts.

When Reddit OAuth app credentials are configured, the primary path is the
authenticated JSON search endpoint (``oauth.reddit.com/r/{sub}/search.json``),
which carries the richest data (score, comment count, body). Reddit's
unauthenticated public JSON endpoint increasingly returns ``HTTP 403 Blocked``
(issue #862), so anonymous runs skip that noisy dead path and use the public
Atom/RSS search feed (``/search.rss``) directly. The RSS fallback lacks score /
comment counts, so RSS-sourced posts are marked and the formatter omits those
metrics rather than printing fake zeros.

No API key is required for RSS. Set ``REDDIT_CLIENT_ID`` and
``REDDIT_CLIENT_SECRET`` to restore JSON score/comment metadata. Returns
formatted plaintext blocks ready for prompt injection and degrades gracefully —
returns a placeholder string rather than raising, so callers never special-case
missing data.
"""

from __future__ import annotations

import base64
import html
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_OAUTH_API = "https://oauth.reddit.com/r/{sub}/search.json?{qs}"
_OAUTH_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_RSS = "https://www.reddit.com/r/{sub}/search.rss?{qs}"
# A descriptive, identified User-Agent (per Reddit's API etiquette). Reddit
# blocks generic/anonymous tokens like bare "Mozilla/5.0" or "curl/…" but
# serves this one on both endpoints; the RSS feed accepts it even when the
# JSON search endpoint 403s, so no browser-spoofing is needed.
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_TOKEN_CACHE: dict[str, float | str] = {}

# Default subreddits ordered roughly by signal density for ticker-specific
# discussion. wallstreetbets has the most volume but most noise; stocks /
# investing trend more measured. Caller can override.
DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")


def _search_qs(ticker: str, limit: int) -> str:
    return urlencode({
        "q": ticker,
        "restrict_sr": "on",
        "sort": "new",
        "t": "week",  # last 7 days
        "limit": limit,
    })


def _user_agent() -> str:
    return os.environ.get("REDDIT_USER_AGENT") or _UA


def _reddit_credentials() -> tuple[str, str] | None:
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    if client_id and client_secret:
        return client_id, client_secret
    return None


def _get_oauth_token(timeout: float) -> str | None:
    """Return a cached Reddit application OAuth token when configured."""
    token = _TOKEN_CACHE.get("access_token")
    expires_at = float(_TOKEN_CACHE.get("expires_at") or 0)
    if isinstance(token, str) and time.time() < expires_at - 60:
        return token

    credentials = _reddit_credentials()
    if not credentials:
        return None

    client_id, client_secret = credentials
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    data = urlencode({"grant_type": "client_credentials"}).encode()
    req = Request(
        _OAUTH_TOKEN_URL,
        data=data,
        headers={
            "Authorization": f"Basic {basic}",
            "User-Agent": _user_agent(),
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except (HTTPError, URLError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning("Reddit OAuth token fetch failed: %s", exc)
        return None

    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        logger.warning("Reddit OAuth token response did not include access_token")
        return None

    expires_in = float(payload.get("expires_in") or 3600)
    _TOKEN_CACHE["access_token"] = access_token
    _TOKEN_CACHE["expires_at"] = time.time() + expires_in
    return access_token


def _iso_to_timestamp(iso_str: Optional[str]) -> Optional[float]:
    """Parse an Atom ``published`` timestamp to a UTC epoch, or None."""
    if not iso_str:
        return None
    try:
        normalized = iso_str[:-1] + "+00:00" if iso_str.endswith("Z") else iso_str
        return datetime.fromisoformat(normalized).timestamp()
    except (ValueError, TypeError):
        return None


def _strip_html(content: str) -> str:
    """Reduce the HTML body Reddit embeds in an Atom entry to plain text."""
    if not content:
        return ""
    # Reddit wraps the real selftext between SC_OFF / SC_ON markers.
    if "<!-- SC_OFF -->" in content and "<!-- SC_ON -->" in content:
        content = content.split("<!-- SC_OFF -->")[1].split("<!-- SC_ON -->")[0]
    text = re.sub(r"<[^>]+>", " ", content)
    return " ".join(html.unescape(text).split())


def _fetch_subreddit_rss(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict]:
    """Fallback path: parse the public Atom search feed for a subreddit.

    Carries no score / comment counts, so those fields are left None and the
    post is tagged ``source="rss"`` for honest display.
    """
    url = _RSS.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(url, headers={"User-Agent": _user_agent()})
    try:
        with urlopen(req, timeout=timeout) as resp:
            root = ET.fromstring(resp.read())
    except (HTTPError, URLError, TimeoutError, ET.ParseError) as exc:
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return []

    posts = []
    for entry in root.findall("atom:entry", _ATOM_NS)[:limit]:
        title_el = entry.find("atom:title", _ATOM_NS)
        published_el = entry.find("atom:published", _ATOM_NS)
        content_el = entry.find("atom:content", _ATOM_NS)
        posts.append({
            "title": (title_el.text if title_el is not None else "") or "",
            "score": None,
            "num_comments": None,
            "created_utc": _iso_to_timestamp(
                published_el.text if published_el is not None else None
            ),
            "selftext": _strip_html(content_el.text if content_el is not None else ""),
            "source": "rss",
        })
    return posts


def _fetch_subreddit(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict]:
    token = _get_oauth_token(timeout)
    if not token:
        logger.debug(
            "Reddit OAuth unavailable; using RSS feed for r/%s · %s.",
            sub,
            ticker,
        )
        return _fetch_subreddit_rss(ticker, sub, limit, timeout)

    url = _OAUTH_API.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": _user_agent(),
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        children = (payload.get("data") or {}).get("children") or []
        return [c.get("data", {}) for c in children if isinstance(c, dict)]
    except (HTTPError, URLError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning(
            "Reddit OAuth JSON fetch failed for r/%s · %s: %s — falling back to RSS feed.",
            sub, ticker, exc,
        )
        return _fetch_subreddit_rss(ticker, sub, limit, timeout)


def fetch_reddit_posts(
    ticker: str,
    subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    inter_request_delay: float = 0.4,
) -> str:
    """Fetch recent Reddit posts mentioning ``ticker`` across finance
    subreddits and return them as a formatted plaintext block.

    ``inter_request_delay`` keeps us under Reddit's public rate limit
    (~10 req/min per IP) even if the caller queries many subreddits.
    """
    blocks = []
    total_posts = 0
    for i, sub in enumerate(subreddits):
        if i > 0:
            time.sleep(inter_request_delay)
        posts = _fetch_subreddit(ticker, sub, limit_per_sub, timeout)
        total_posts += len(posts)
        if not posts:
            blocks.append(f"r/{sub}: <no posts found mentioning {ticker.upper()} in the past 7 days>")
            continue

        via_rss = any(p.get("source") == "rss" for p in posts)
        header = f"r/{sub} — {len(posts)} recent posts mentioning {ticker.upper()}"
        header += " (via RSS feed; scores/comments unavailable):" if via_rss else ":"
        lines = [header]
        for p in posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            score = p.get("score")
            comments = p.get("num_comments")
            created = p.get("created_utc")
            created_str = (
                time.strftime("%Y-%m-%d", time.gmtime(created)) if created else "?"
            )
            # Score / comment counts are absent on the RSS fallback path —
            # show them only when present rather than printing fake zeros.
            meta = created_str
            if score is not None and comments is not None:
                meta += f" · {score:>4}↑ · {comments:>3}c"
            selftext = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(selftext) > 240:
                selftext = selftext[:240] + "…"
            lines.append(
                f"  [{meta}] {title}"
                + (f"\n    body excerpt: {selftext}" if selftext else "")
            )
        blocks.append("\n".join(lines))

    if total_posts == 0:
        return (
            f"<no Reddit posts found mentioning {ticker.upper()} across "
            f"{', '.join(f'r/{s}' for s in subreddits)} in the past 7 days>"
        )
    return "\n\n".join(blocks)
