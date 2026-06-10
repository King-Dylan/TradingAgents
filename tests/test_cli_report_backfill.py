from __future__ import annotations

from cli.main import backfill_empty_analyst_reports


def test_backfills_empty_streamed_analyst_report_from_display_buffer():
    final_state = {
        "market_report": "market",
        "sentiment_report": "sentiment",
        "news_report": "",
        "fundamentals_report": "fundamentals",
    }
    report_sections = {
        "market_report": "market from display",
        "sentiment_report": "sentiment from display",
        "news_report": "news from display",
        "fundamentals_report": "fundamentals from display",
    }

    result = backfill_empty_analyst_reports(final_state, report_sections)

    assert result["market_report"] == "market"
    assert result["news_report"] == "news from display"


def test_backfill_ignores_blank_display_sections():
    final_state = {"news_report": ""}
    report_sections = {"news_report": "   "}

    result = backfill_empty_analyst_reports(final_state, report_sections)

    assert result["news_report"] == ""
