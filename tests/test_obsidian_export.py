from __future__ import annotations

import datetime as dt
from pathlib import Path

from tradingagents.obsidian_export import export_report_to_obsidian


def _write_source_report(tmp_path: Path, run_id: str = "NBIS_codex_full_20260608_220338") -> Path:
    report_dir = tmp_path / "reports" / run_id
    report_dir.mkdir(parents=True)
    report_file = report_dir / "complete_report.md"
    report_file.write_text(
        "# Trading Analysis Report: NBIS\n\n"
        "Generated: 2026-06-08 22:03:38\n\n"
        "## V. Portfolio Manager Decision\n\n"
        "Final decision body.",
        encoding="utf-8",
    )
    return report_file


def _final_state(target: str = "300.0") -> dict:
    return {
        "trader_investment_plan": (
            "**Action**: Buy\n\n"
            "**Reasoning**: Momentum and fundamentals line up.\n\n"
            "**Entry Price**: 218.0\n\n"
            "**Stop Loss**: 172.0\n\n"
            "**Position Sizing**: 2-4% satellite position; first tranche 40-50%."
        ),
        "final_trade_decision": (
            "**Rating**: Overweight\n\n"
            "**Executive Summary**: Build gradually with defined risk.\n\n"
            "**Investment Thesis**: Evidence favors a constructive long-term view.\n\n"
            f"**Price Target**: {target}\n\n"
            "**Time Horizon**: 12-24 months"
        ),
    }


def _config(vault_dir: Path) -> dict:
    return {
        "obsidian_auto_export": True,
        "obsidian_vault_dir": str(vault_dir),
        "obsidian_tradingagents_dir": "TradingAgents",
        "obsidian_summary_page": "股票分析总览.md",
    }


def test_export_report_to_obsidian_writes_report_decision_summary_and_indexes(tmp_path):
    report_file = _write_source_report(tmp_path)
    vault_dir = tmp_path / "vault"

    result = export_report_to_obsidian(
        _final_state(),
        "NBIS",
        report_file,
        _config(vault_dir),
        now=dt.datetime(2026, 6, 9, 15, 0, 0),
    )

    assert result is not None
    assert result.report_path.read_text(encoding="utf-8").startswith("---\ntype: \"tradingagents_report\"")
    assert "**Rating**: Overweight" in result.decision_path.read_text(encoding="utf-8")

    summary = result.summary_path.read_text(encoding="utf-8")
    assert "| [[TradingAgents/Stocks/NBIS\\|NBIS]] | 2026-06-08 22:03:38 | codex_full | Overweight | Buy |" in summary
    assert "[[TradingAgents/Reports/NBIS_codex_full_20260608_220338\\|完整报告]]" in summary
    assert "218.0" in summary
    assert "172.0" in summary
    assert "300.0" in summary
    assert "12-24 months" in summary

    stock_page = result.stock_path.read_text(encoding="utf-8")
    assert "# NBIS 分析记录" in stock_page
    assert "- 最新报告：[[TradingAgents/Reports/NBIS_codex_full_20260608_220338|完整报告]]" in stock_page

    reports_index = (vault_dir / "TradingAgents" / "Reports.md").read_text(encoding="utf-8")
    decisions_index = (vault_dir / "TradingAgents" / "Decisions.md").read_text(encoding="utf-8")
    stocks_index = (vault_dir / "TradingAgents" / "Stocks.md").read_text(encoding="utf-8")
    assert "[[TradingAgents/Reports/NBIS_codex_full_20260608_220338|NBIS_codex_full_20260608_220338]]" in reports_index
    assert "[[TradingAgents/Decisions/NBIS_codex_full_20260608_220338 - PM Decision|NBIS_codex_full_20260608_220338 - PM Decision]]" in decisions_index
    assert "[[TradingAgents/Stocks/NBIS|NBIS]]" in stocks_index


def test_export_report_to_obsidian_replaces_existing_run_in_tables(tmp_path):
    report_file = _write_source_report(tmp_path)
    vault_dir = tmp_path / "vault"
    config = _config(vault_dir)
    now = dt.datetime(2026, 6, 9, 15, 0, 0)

    first = export_report_to_obsidian(_final_state("300.0"), "NBIS", report_file, config, now=now)
    second = export_report_to_obsidian(_final_state("305.0"), "NBIS", report_file, config, now=now)

    assert first is not None
    assert second is not None
    summary_lines = second.summary_path.read_text(encoding="utf-8").splitlines()
    rows_for_run = [
        line for line in summary_lines
        if line.startswith("| [[TradingAgents/Stocks/NBIS") and "NBIS_codex_full_20260608_220338" in line
    ]
    assert len(rows_for_run) == 2
    assert all("305.0" in line for line in rows_for_run)


def test_export_report_to_obsidian_disabled_is_noop(tmp_path):
    report_file = _write_source_report(tmp_path)
    result = export_report_to_obsidian(
        _final_state(),
        "NBIS",
        report_file,
        {"obsidian_auto_export": False},
    )
    assert result is None
