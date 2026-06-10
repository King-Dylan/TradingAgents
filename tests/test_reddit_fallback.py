"""Tests for Reddit OAuth JSON and RSS/Atom fallback handling (#862)."""

from __future__ import annotations

import json
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from tradingagents.dataflows import reddit


_SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>NVDA earnings beat, stock pops</title>
    <published>2026-05-20T14:30:00+00:00</published>
    <content type="html">&lt;!-- SC_OFF --&gt;&lt;div class="md"&gt;&lt;p&gt;Great &lt;b&gt;quarter&lt;/b&gt; for NVDA&amp;#39;s datacenter unit.&lt;/p&gt;&lt;/div&gt;&lt;!-- SC_ON --&gt;</content>
  </entry>
  <entry>
    <title>Is NVDA overvalued?</title>
    <published>2026-05-19T09:00:00Z</published>
    <content type="html">&lt;p&gt;Forward P/E discussion&lt;/p&gt;</content>
  </entry>
</feed>
"""


@pytest.mark.unit
class TestIsoToTimestamp:
    def test_parses_offset_and_z(self):
        assert reddit._iso_to_timestamp("2026-05-20T14:30:00+00:00") > 0
        assert reddit._iso_to_timestamp("2026-05-19T09:00:00Z") > 0

    def test_none_and_garbage_return_none(self):
        assert reddit._iso_to_timestamp(None) is None
        assert reddit._iso_to_timestamp("not-a-date") is None


@pytest.mark.unit
class TestStripHtml:
    def test_extracts_between_sc_markers_and_unescapes(self):
        raw = "<!-- SC_OFF --><div class=\"md\"><p>Great <b>quarter</b> &amp; more</p></div><!-- SC_ON -->"
        assert reddit._strip_html(raw) == "Great quarter & more"

    def test_empty(self):
        assert reddit._strip_html("") == ""


@pytest.mark.unit
class TestRssFallbackParsing:
    def _patch_rss_response(self, xml_bytes):
        class _Resp:
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, *a):
                return False
            def read(self_inner):
                return xml_bytes
        return patch.object(reddit, "urlopen", return_value=_Resp())

    def test_parses_atom_entries(self):
        with self._patch_rss_response(_SAMPLE_ATOM.encode("utf-8")):
            posts = reddit._fetch_subreddit_rss("NVDA", "stocks", limit=5, timeout=5.0)
        assert len(posts) == 2
        assert posts[0]["title"] == "NVDA earnings beat, stock pops"
        assert posts[0]["source"] == "rss"
        assert posts[0]["score"] is None
        assert posts[0]["num_comments"] is None
        assert posts[0]["created_utc"] > 0
        assert "datacenter unit" in posts[0]["selftext"]

    def test_malformed_xml_fails_open(self):
        with self._patch_rss_response(b"<<not xml>>"):
            assert reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0) == []


@pytest.mark.unit
class TestJsonFallsBackToRss:
    def setup_method(self):
        reddit._TOKEN_CACHE.clear()

    def test_without_oauth_credentials_uses_rss_directly(self, monkeypatch):
        monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
        monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
        with patch.object(reddit, "urlopen") as urlopen, \
             patch.object(reddit, "_fetch_subreddit_rss", return_value=[{"title": "x", "source": "rss", "score": None, "num_comments": None, "created_utc": None, "selftext": ""}]) as rss:
            out = reddit._fetch_subreddit("NVDA", "stocks", 5, 5.0)
        urlopen.assert_not_called()
        rss.assert_called_once()
        assert out and out[0]["source"] == "rss"

    def test_oauth_403_triggers_rss(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "client")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
        reddit._TOKEN_CACHE["access_token"] = "token"
        reddit._TOKEN_CACHE["expires_at"] = reddit.time.time() + 3600
        err = HTTPError("url", 403, "Blocked", {}, None)
        with patch.object(reddit, "urlopen", side_effect=err), \
             patch.object(reddit, "_fetch_subreddit_rss", return_value=[{"title": "x", "source": "rss", "score": None, "num_comments": None, "created_utc": None, "selftext": ""}]) as rss:
            out = reddit._fetch_subreddit("NVDA", "stocks", 5, 5.0)
        rss.assert_called_once()
        assert out and out[0]["source"] == "rss"

    def test_oauth_json_search_returns_posts(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "client")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")

        class _Resp:
            def __init__(self, payload):
                self.payload = payload
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        token_payload = {"access_token": "token", "expires_in": 3600}
        search_payload = {
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "NVDA JSON post",
                            "score": 42,
                            "num_comments": 7,
                            "created_utc": 1770000000,
                            "selftext": "body",
                        }
                    }
                ]
            }
        }
        with patch.object(reddit, "urlopen", side_effect=[_Resp(token_payload), _Resp(search_payload)]) as urlopen:
            out = reddit._fetch_subreddit("NVDA", "stocks", 5, 5.0)
        assert urlopen.call_count == 2
        assert out[0]["title"] == "NVDA JSON post"
        assert out[0]["score"] == 42


@pytest.mark.unit
class TestFormatterHandlesRssPosts:
    def test_rss_posts_omit_fake_counts_and_note_source(self):
        rss_posts = [{
            "title": "NVDA pops", "score": None, "num_comments": None,
            "created_utc": reddit._iso_to_timestamp("2026-05-20T14:30:00Z"),
            "selftext": "great quarter", "source": "rss",
        }]
        with patch.object(reddit, "_fetch_subreddit", return_value=rss_posts):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",), inter_request_delay=0)
        assert "via RSS feed" in out
        assert "↑" not in out  # no fake score arrow
        assert "NVDA pops" in out
        assert "great quarter" in out

    def test_json_posts_still_show_counts(self):
        json_posts = [{
            "title": "NVDA pops", "score": 1234, "num_comments": 56,
            "created_utc": reddit._iso_to_timestamp("2026-05-20T14:30:00Z"),
            "selftext": "",
        }]
        with patch.object(reddit, "_fetch_subreddit", return_value=json_posts):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",), inter_request_delay=0)
        assert "1234↑" in out
        assert "56c" in out
        assert "via RSS" not in out
