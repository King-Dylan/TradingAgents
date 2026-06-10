from __future__ import annotations

from pathlib import Path


ANALYST_FILES = (
    Path("tradingagents/agents/analysts/market_analyst.py"),
    Path("tradingagents/agents/analysts/fundamentals_analyst.py"),
    Path("tradingagents/agents/analysts/news_analyst.py"),
    Path("tradingagents/agents/analysts/sentiment_analyst.py"),
)


def test_analyst_prompts_do_not_invite_trader_stop_signal():
    for path in ANALYST_FILES:
        source = path.read_text(encoding="utf-8")
        assert "prefix your response with FINAL TRANSACTION PROPOSAL" not in source
        assert "Do not output a portfolio rating or FINAL TRANSACTION PROPOSAL" in source
