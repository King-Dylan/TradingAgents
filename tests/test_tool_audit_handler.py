import json

from cli.stats_handler import ToolAuditCallbackHandler


def test_tool_audit_handler_logs_success_with_calendar_gaps(tmp_path):
    log_path = tmp_path / "tool_audit.jsonl"
    handler = ToolAuditCallbackHandler(log_path)

    handler.on_tool_start(
        {"name": "get_indicators"},
        '{"symbol":"NBIS","indicator":"rsi"}',
        run_id="run-1",
    )
    handler.on_tool_end(
        "2026-06-08: 55.1\n2026-06-07: N/A: Not a trading day (weekend or holiday)",
        run_id="run-1",
    )

    summary = handler.get_summary()
    assert summary["total"] == 1
    assert summary["failure_count"] == 0
    assert summary["by_status"] == {"ok_with_calendar_gaps": 1}

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["tool_start", "tool_end"]
    assert records[-1]["status"] == "ok_with_calendar_gaps"
    assert records[-1]["tool"] == "get_indicators"


def test_tool_audit_handler_classifies_configuration_and_rate_limit_failures():
    handler = ToolAuditCallbackHandler()

    handler.on_tool_start({"name": "get_news"}, "{}", run_id="run-2")
    handler.on_tool_end("Error fetching news: API key is not configured", run_id="run-2")

    handler.on_tool_start({"name": "get_stock_data"}, "{}", run_id="run-3")
    handler.on_tool_end("Alpha Vantage rate limit exceeded: 429", run_id="run-3")

    summary = handler.get_summary()
    assert summary["by_status"] == {"not_configured": 1, "rate_limited": 1}
    assert summary["failure_count"] == 2
    assert [failure["tool"] for failure in summary["failures"]] == [
        "get_news",
        "get_stock_data",
    ]


def test_tool_audit_handler_does_not_treat_plain_financial_429_as_rate_limit():
    handler = ToolAuditCallbackHandler()

    handler.on_tool_start({"name": "get_balance_sheet"}, "{}", run_id="run-plain-429")
    handler.on_tool_end(
        "# Balance Sheet\nOrdinary Shares Number,875553173.0,847259927.0",
        run_id="run-plain-429",
    )

    summary = handler.get_summary()
    assert summary["by_status"] == {"ok": 1}
    assert summary["failure_count"] == 0


def test_tool_audit_handler_records_tool_exceptions():
    handler = ToolAuditCallbackHandler()

    handler.on_tool_start({"name": "get_fundamentals"}, "{}", run_id="run-4")
    handler.on_tool_error(RuntimeError("network down"), run_id="run-4")

    summary = handler.get_summary()
    assert summary["by_status"] == {"exception": 1}
    assert summary["failure_count"] == 1
    assert "network down" in summary["failures"][0]["reason"]
