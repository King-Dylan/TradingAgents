#!/usr/bin/env python3
"""Run Codex-backed TradingAgents regression cases and compare to baselines."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from cli.main import save_report_to_disk
from cli.stats_handler import StatsCallbackHandler, ToolAuditCallbackHandler
from tradingagents.agents.utils.rating import parse_rating
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.obsidian_export import export_report_to_obsidian


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
ANALYSTS = ["market", "social", "news", "fundamentals"]
RUNNER_ERROR_DETAIL_LIMIT = 2000


@dataclass(frozen=True)
class RegressionCase:
    ticker: str
    trade_date: str
    baseline_run: str
    strict_outcome: bool
    note: str


CASES: dict[str, RegressionCase] = {
    "NBIS": RegressionCase("NBIS", "2026-06-08", "NBIS_20260608_150245", True, "old API golden buy/overweight case"),
    "QCOM": RegressionCase("QCOM", "2026-06-08", "QCOM_20260608_155744", True, "old API golden overweight/buy case"),
    "MRVL": RegressionCase("MRVL", "2026-06-08", "MRVL_20260608_182628", True, "old API negative golden hold/hold case"),
    "NXPI": RegressionCase("NXPI", "2026-06-08", "NXPI_20260608_153106", True, "old API negative golden hold/hold case"),
    "PLTR": RegressionCase("PLTR", "2026-06-09", "PLTR_20260610_090502", False, "latest local PLTR report; checks transcript/report-shape regression"),
    "AMAT": RegressionCase("AMAT", "2026-06-09", "AMAT_20260609_142308", False, "local report health case; no strict API baseline"),
    "ENVX": RegressionCase("ENVX", "2026-06-08", "ENVX_20260609_000841", False, "local report health case; no strict API baseline"),
    "NOK": RegressionCase("NOK", "2026-06-08", "NOK_20260609_001543", False, "local report health case; no strict API baseline"),
}


def main() -> int:
    args = parse_args()
    selected_cases = resolve_cases(args.cases)
    batch_id = args.batch_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = REPORTS_DIR / f"codex_regression_batch_{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    safe_print(f"[batch] id={batch_id} cases={','.join(case.ticker for case in selected_cases)}")
    safe_print(f"[batch] output={batch_dir}")

    results: list[dict[str, Any]] = []
    for index, case in enumerate(selected_cases, start=1):
        safe_print(f"[case {index}/{len(selected_cases)}] start {case.ticker} {case.trade_date}")
        started_at = time.time()
        try:
            result = run_case(case, batch_id=batch_id, args=args)
        except Exception as exc:  # noqa: BLE001 - keep batch running
            run_dir = REPORTS_DIR / f"{case.ticker}_codex_latest_{compact_date(case.trade_date)}_{batch_id}"
            result = {
                "ticker": case.ticker,
                "trade_date": case.trade_date,
                "baseline_run": case.baseline_run,
                "error": format_exception(exc),
                "elapsed_seconds": round(time.time() - started_at, 2),
                "run_dir": str(run_dir),
                "progress_log": str(run_dir / "progress.log"),
                "tool_audit_log": str(run_dir / "tool_audit.jsonl"),
                "status": "ERROR",
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(run_dir / "run_summary.json", result)
            safe_print(f"[case {case.ticker}] ERROR {result['error']}")
        results.append(result)
        write_json(batch_dir / "batch_results.json", results)
        write_batch_markdown(batch_dir / "batch_summary.md", results)
        safe_print(
            f"[case {case.ticker}] done status={result.get('status')} "
            f"elapsed={result.get('elapsed_seconds')}s report={result.get('report_file', '-')}"
        )

    write_json(batch_dir / "batch_results.json", results)
    write_batch_markdown(batch_dir / "batch_summary.md", results)
    safe_print(f"[batch] complete summary={batch_dir / 'batch_summary.md'}")
    return 0 if all(result.get("status") == "PASS" for result in results) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="golden,PLTR",
        help="Comma-separated tickers, or one of: golden, health, all. Default: golden,PLTR",
    )
    parser.add_argument("--batch-id", help="Stable id for output paths")
    parser.add_argument(
        "--reasoning-effort",
        default="xhigh",
        help="Codex reasoning effort for all runs. Default: xhigh",
    )
    parser.add_argument(
        "--codex-timeout",
        type=int,
        default=7200,
        help="Per Codex LLM-call timeout in seconds. Default: 7200",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Forwarded sampling temperature when provider honors it. Default: 0.0",
    )
    parsed = parser.parse_args()
    parsed.selected_reasoning_effort = parsed.reasoning_effort
    return parsed


def resolve_cases(raw: str) -> list[RegressionCase]:
    selected: list[RegressionCase] = []
    for token in (part.strip().upper() for part in raw.split(",")):
        if not token:
            continue
        if token == "GOLDEN":
            selected.extend(CASES[key] for key in ("NBIS", "QCOM", "MRVL", "NXPI"))
        elif token == "HEALTH":
            selected.extend(CASES[key] for key in ("PLTR", "AMAT", "ENVX", "NOK"))
        elif token == "ALL":
            selected.extend(CASES[key] for key in CASES)
        elif token in CASES:
            selected.append(CASES[token])
        else:
            raise SystemExit(f"Unknown case {token!r}. Known: {', '.join(CASES)}")
    deduped: list[RegressionCase] = []
    seen: set[str] = set()
    for case in selected:
        if case.ticker not in seen:
            deduped.append(case)
            seen.add(case.ticker)
    return deduped


def run_case(case: RegressionCase, *, batch_id: str, args: argparse.Namespace) -> dict[str, Any]:
    run_dir = REPORTS_DIR / f"{case.ticker}_codex_latest_{compact_date(case.trade_date)}_{batch_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_log = run_dir / "progress.log"
    tool_audit_file = run_dir / "tool_audit.jsonl"

    stats_handler = StatsCallbackHandler()
    tool_audit_handler = ToolAuditCallbackHandler(tool_audit_file)
    callbacks = [stats_handler, tool_audit_handler]

    config = DEFAULT_CONFIG.copy()
    config.update(
        {
            "llm_provider": "codex",
            "quick_think_llm": "default",
            "deep_think_llm": "default",
            "codex_reasoning_effort": args.selected_reasoning_effort,
            "codex_timeout": args.codex_timeout,
            "temperature": args.temperature,
            "max_debate_rounds": 5,
            "max_risk_discuss_rounds": 5,
            "output_language": "Chinese",
            "checkpoint_enabled": False,
            "memory_log_path": str(run_dir / "isolated_memory.md"),
            "results_dir": str(run_dir / "runtime_logs"),
        }
    )

    graph = TradingAgentsGraph(ANALYSTS, config=config, debug=False, callbacks=callbacks)
    instrument_context = graph.resolve_instrument_context(case.ticker, "stock")
    init_state = graph.propagator.create_initial_state(
        case.ticker,
        case.trade_date,
        asset_type="stock",
        past_context="",
        instrument_context=instrument_context,
    )
    args = graph.propagator.get_graph_args(callbacks=callbacks)

    started_at = time.time()
    final_state: dict[str, Any] = {}
    last_progress: dict[str, Any] = {}
    for chunk_index, chunk in enumerate(graph.graph.stream(init_state, **args), start=1):
        final_state.update(chunk)
        progress = progress_snapshot(final_state)
        if progress != last_progress:
            line = (
                f"{dt.datetime.now().isoformat(timespec='seconds')} "
                f"chunk={chunk_index} {format_progress(progress)}"
            )
            append_line(progress_log, line)
            safe_print(f"[case {case.ticker}] {format_progress(progress)}")
            last_progress = progress

    elapsed = round(time.time() - started_at, 2)
    report_file = save_report_to_disk(final_state, case.ticker, run_dir)
    obsidian = export_report_to_obsidian(final_state, case.ticker, report_file, config)

    extracted = extract_report_fields(run_dir)
    baseline_dir = REPORTS_DIR / case.baseline_run
    baseline = extract_report_fields(baseline_dir) if baseline_dir.exists() else {}
    comparison = compare_to_baseline(extracted, baseline, strict=case.strict_outcome)
    section_chars = collect_section_chars(run_dir)
    tool_audit = tool_audit_handler.get_summary()
    stats = stats_handler.get_stats()

    result = {
        "ticker": case.ticker,
        "trade_date": case.trade_date,
        "baseline_run": case.baseline_run,
        "note": case.note,
        "strict_outcome": case.strict_outcome,
        "run_dir": str(run_dir),
        "report_file": str(report_file),
        "obsidian_report": str(obsidian.report_path) if obsidian else None,
        "elapsed_seconds": elapsed,
        "config": {
            "llm_provider": config["llm_provider"],
            "quick_think_llm": config["quick_think_llm"],
            "deep_think_llm": config["deep_think_llm"],
            "codex_reasoning_effort": config["codex_reasoning_effort"],
            "max_debate_rounds": config["max_debate_rounds"],
            "max_risk_discuss_rounds": config["max_risk_discuss_rounds"],
            "output_language": config["output_language"],
            "temperature": config["temperature"],
        },
        "stats": stats,
        "tool_audit": tool_audit,
        "section_chars": section_chars,
        "extracted": extracted,
        "baseline": baseline,
        "comparison": comparison,
        "status": status_from_result(tool_audit, comparison),
    }
    write_json(run_dir / "run_summary.json", result)
    return result


def progress_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    debate = state.get("investment_debate_state") or {}
    risk = state.get("risk_debate_state") or {}
    return {
        "analysts": ",".join(
            name
            for name, key in (
                ("market", "market_report"),
                ("social", "sentiment_report"),
                ("news", "news_report"),
                ("fundamentals", "fundamentals_report"),
            )
            if state.get(key)
        ),
        "research_count": debate.get("count", 0),
        "research_manager": bool(debate.get("judge_decision")),
        "trader": bool(state.get("trader_investment_plan")),
        "risk_count": risk.get("count", 0),
        "pm": bool(risk.get("judge_decision") or state.get("final_trade_decision")),
    }


def format_progress(progress: dict[str, Any]) -> str:
    return (
        f"analysts=[{progress['analysts']}] "
        f"research={progress['research_count']} manager={int(progress['research_manager'])} "
        f"trader={int(progress['trader'])} "
        f"risk={progress['risk_count']} pm={int(progress['pm'])}"
    )


def extract_report_fields(report_dir: Path) -> dict[str, Any]:
    pm_text = read_text(report_dir / "5_portfolio" / "decision.md")
    trader_text = read_text(report_dir / "3_trading" / "trader.md")
    manager_text = read_text(report_dir / "2_research" / "manager.md")
    combined_decision_context = "\n\n".join(part for part in (trader_text, pm_text) if part)
    research_debate = read_text(report_dir / "2_research" / "debate.md")
    risk_debate = read_text(report_dir / "4_risk" / "debate.md")
    complete_report = read_text(report_dir / "complete_report.md")

    return {
        "manager_rating": parse_rating(manager_text, default="-") if manager_text else "-",
        "trader_action": extract_action(trader_text, pm_text),
        "pm_rating": parse_rating(pm_text, default="-") if pm_text else "-",
        "entry_price": extract_first_label(combined_decision_context, ("Entry Price", "Entry", "买入价", "入场价")),
        "stop_loss": extract_stop_loss(combined_decision_context),
        "position_sizing": extract_position_sizing(combined_decision_context),
        "price_target": extract_first_label(pm_text, ("Price Target", "Target Price", "Target", "目标价")),
        "time_horizon": extract_first_label(pm_text, ("Time Horizon", "Holding Period", "周期", "时间框架")),
        "research_debate_transcript": bool(research_debate),
        "risk_debate_transcript": bool(risk_debate),
        "complete_has_research_transcript": "Bull/Bear Debate Transcript" in complete_report,
        "complete_has_risk_transcript": "Risk Debate Transcript" in complete_report,
        "bull_rounds": count_agent_turns(report_dir / "2_research" / "bull.md", "Bull Analyst:"),
        "bear_rounds": count_agent_turns(report_dir / "2_research" / "bear.md", "Bear Analyst:"),
        "aggressive_rounds": count_agent_turns(report_dir / "4_risk" / "aggressive.md", "Aggressive Analyst:"),
        "conservative_rounds": count_agent_turns(report_dir / "4_risk" / "conservative.md", "Conservative Analyst:"),
        "neutral_rounds": count_agent_turns(report_dir / "4_risk" / "neutral.md", "Neutral Analyst:"),
        "research_scorecard_markers": count_scorecard_markers(read_text(report_dir / "2_research" / "bull.md"))
        + count_scorecard_markers(read_text(report_dir / "2_research" / "bear.md")),
        "risk_scorecard_markers": count_scorecard_markers(read_text(report_dir / "4_risk" / "aggressive.md"))
        + count_scorecard_markers(read_text(report_dir / "4_risk" / "conservative.md"))
        + count_scorecard_markers(read_text(report_dir / "4_risk" / "neutral.md")),
        "complete_chars": len(complete_report),
    }


def compare_to_baseline(current: dict[str, Any], baseline: dict[str, Any], *, strict: bool) -> dict[str, Any]:
    transcript_ok = (
        current.get("research_debate_transcript")
        and current.get("risk_debate_transcript")
        and current.get("complete_has_research_transcript")
        and current.get("complete_has_risk_transcript")
    )
    round_ok = all(
        current.get(field, 0) >= expected
        for field, expected in (
            ("bull_rounds", 5),
            ("bear_rounds", 5),
            ("aggressive_rounds", 5),
            ("conservative_rounds", 5),
            ("neutral_rounds", 5),
        )
    )
    if not baseline:
        return {
            "strict": strict,
            "available": False,
            "outcome_match": None,
            "key_field_match": None,
            "mismatches": [],
            "key_field_mismatches": [],
            "transcript_ok": transcript_ok,
            "round_ok": round_ok,
        }
    fields = ["manager_rating", "trader_action", "pm_rating"]
    mismatches = [
        {"field": field, "baseline": baseline.get(field), "current": current.get(field)}
        for field in fields
        if baseline.get(field) != current.get(field)
    ]
    key_field_mismatches = compare_key_fields(current, baseline)
    return {
        "strict": strict,
        "available": True,
        "outcome_match": not mismatches,
        "mismatches": mismatches,
        "key_field_match": not key_field_mismatches,
        "key_field_mismatches": key_field_mismatches,
        "transcript_ok": transcript_ok,
        "round_ok": round_ok,
    }


def compare_key_fields(current: dict[str, Any], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for field, tolerance in (
        ("price_target", 0.05),
        ("entry_price", 0.03),
        ("stop_loss", 0.03),
    ):
        baseline_value = baseline.get(field)
        current_value = current.get(field)
        if not has_value(baseline_value):
            continue
        if not numeric_values_close(current_value, baseline_value, tolerance=tolerance):
            mismatches.append({
                "field": field,
                "baseline": baseline_value,
                "current": current_value,
                "reason": f"numeric drift > {int(tolerance * 100)}%",
            })

    for field in ("time_horizon", "position_sizing"):
        baseline_value = baseline.get(field)
        current_value = current.get(field)
        if not has_value(baseline_value):
            continue
        if not text_key_field_matches(field, current_value, baseline_value):
            mismatches.append({
                "field": field,
                "baseline": baseline_value,
                "current": current_value,
                "reason": "text contract drift",
            })
    return mismatches


def has_value(value: Any) -> bool:
    return value is not None and str(value).strip() not in {"", "-"}


def numeric_values_close(current: Any, baseline: Any, *, tolerance: float) -> bool:
    current_number = first_number(current)
    baseline_number = first_number(baseline)
    if current_number is None or baseline_number is None:
        return clean_for_compare(current) == clean_for_compare(baseline)
    allowed = max(5.0, abs(baseline_number) * tolerance)
    return abs(current_number - baseline_number) <= allowed


def first_number(value: Any) -> float | None:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else None


def text_key_field_matches(field: str, current: Any, baseline: Any) -> bool:
    current_text = clean_for_compare(current)
    baseline_text = clean_for_compare(baseline)
    if not baseline_text or baseline_text == "-":
        return True
    if baseline_text in current_text or current_text in baseline_text:
        return True
    if field == "time_horizon":
        return extracted_ranges(current_text) == extracted_ranges(baseline_text)
    if field == "position_sizing":
        baseline_tokens = important_position_tokens(baseline_text)
        if not baseline_tokens:
            return True
        current_tokens = important_position_tokens(current_text)
        overlap = len(baseline_tokens & current_tokens)
        return overlap / len(baseline_tokens) >= 0.6
    return False


def clean_for_compare(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", "", text.lower())


def extracted_ranges(text: str) -> set[str]:
    return set(re.findall(r"\d+\s*-\s*\d+\s*(?:个?月|months?|年|years?)", text, re.I))


def important_position_tokens(text: str) -> set[str]:
    normalized = re.sub(r"(\d+(?:\.\d+)?)%\s*-\s*(\d+(?:\.\d+)?)%", r"\1-\2%", text)
    tokens = set(re.findall(r"\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?%?|\d+(?:\.\d+)?%", normalized))
    if "sma" in text.lower() or "均线" in text or "50日" in text:
        tokens.add("50sma")
    if "分批" in text or "tranche" in text.lower():
        tokens.add("tranche")
    return {re.sub(r"\s+", "", token.lower()) for token in tokens if token.strip()}


def status_from_result(tool_audit: dict[str, Any], comparison: dict[str, Any]) -> str:
    if tool_audit.get("failure_count", 0):
        return "FAIL_TOOL"
    if comparison.get("strict") and not comparison.get("outcome_match"):
        return "FAIL_OUTCOME"
    if comparison.get("strict") and not comparison.get("key_field_match", True):
        return "FAIL_KEY_FIELDS"
    if not comparison.get("transcript_ok", True):
        return "FAIL_TRANSCRIPT"
    if not comparison.get("round_ok", True):
        return "FAIL_ROUNDS"
    return "PASS"


def write_batch_markdown(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Codex Regression Batch",
        "",
        f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Ticker | Status | Baseline | Manager | Trader | PM | Target | Horizon | Tool audit | Transcript | Report |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        extracted = result.get("extracted", {})
        baseline = result.get("baseline", {})
        comparison = result.get("comparison", {})
        tool_audit = result.get("tool_audit", {})
        transcript = "ok" if comparison.get("transcript_ok") else "missing"
        report = result.get("report_file")
        report_link = f"[report]({report})" if report else "-"
        lines.append(
            "| {ticker} | {status} | {baseline_pm}/{baseline_action} | "
            "{manager} | {trader} | {pm} | {target} | {horizon} | "
            "{tool_total}/{tool_failures} failures | {transcript} | {report} |".format(
                ticker=result.get("ticker", "-"),
                status=result.get("status", "-"),
                baseline_pm=baseline.get("pm_rating", "-"),
                baseline_action=baseline.get("trader_action", "-"),
                manager=extracted.get("manager_rating", "-"),
                trader=extracted.get("trader_action", "-"),
                pm=extracted.get("pm_rating", "-"),
                target=extracted.get("price_target", "-"),
                horizon=extracted.get("time_horizon", "-"),
                tool_total=tool_audit.get("total", "-"),
                tool_failures=tool_audit.get("failure_count", "-"),
                transcript=transcript,
                report=report_link,
            )
        )
    lines.extend(["", "## Mismatches", ""])
    for result in results:
        mismatches = (result.get("comparison") or {}).get("mismatches") or []
        key_mismatches = (result.get("comparison") or {}).get("key_field_mismatches") or []
        if not mismatches:
            mismatches = []
        if not mismatches and not key_mismatches:
            continue
        lines.append(f"### {result.get('ticker')}")
        for mismatch in mismatches:
            lines.append(
                f"- {mismatch['field']}: baseline={mismatch['baseline']} current={mismatch['current']}"
            )
        for mismatch in key_mismatches:
            lines.append(
                f"- {mismatch['field']} ({mismatch.get('reason', 'key drift')}): "
                f"baseline={mismatch['baseline']} current={mismatch['current']}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_section_chars(run_dir: Path) -> dict[str, int]:
    sections: dict[str, int] = {}
    for relative in (
        "1_analysts/market.md",
        "1_analysts/sentiment.md",
        "1_analysts/news.md",
        "1_analysts/fundamentals.md",
        "2_research/debate.md",
        "2_research/bull.md",
        "2_research/bear.md",
        "2_research/manager.md",
        "3_trading/trader.md",
        "4_risk/debate.md",
        "4_risk/aggressive.md",
        "4_risk/conservative.md",
        "4_risk/neutral.md",
        "5_portfolio/decision.md",
        "complete_report.md",
    ):
        path = run_dir / relative
        if path.exists():
            sections[relative] = len(path.read_text(encoding="utf-8"))
    return sections


def extract_action(trader_text: str, pm_text: str) -> str:
    action = extract_first_label(trader_text, ("Action", "交易动作", "交易建议"))
    if action != "-":
        return canonical_action(action)
    match = re.search(r"FINAL TRANSACTION PROPOSAL:\s*\**\s*(BUY|HOLD|SELL)", trader_text, re.I)
    if match:
        return canonical_action(match.group(1))
    rating = parse_rating(pm_text, default="-")
    if rating in {"Buy", "Overweight"}:
        return "Buy"
    if rating in {"Sell", "Underweight"}:
        return "Sell"
    if rating == "Hold":
        return "Hold"
    return "-"


def canonical_action(raw: str) -> str:
    clean = raw.strip().strip("*").split()[0].strip(".,;:").lower()
    return {"buy": "Buy", "hold": "Hold", "sell": "Sell"}.get(clean, raw.strip())


def extract_first_label(text: str, labels: Iterable[str]) -> str:
    if not text:
        return "-"
    for label in labels:
        pattern = re.compile(
            rf"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?\**\s*{re.escape(label)}\s*\**\s*[:：]\s*(.+?)\s*$",
            re.I | re.M,
        )
        match = pattern.search(text)
        if match:
            value = clean_value(match.group(1))
            if value:
                return value
    return "-"


def extract_position_sizing(text: str) -> str:
    labeled = extract_first_label(text, ("Position Sizing", "Position Size", "仓位", "仓位建议", "阶段买入/仓位"))
    if labeled != "-":
        return labeled
    return extract_sentence(text, ("仓位", "分批", "加仓", "position", "sizing")) or "-"


def extract_stop_loss(text: str) -> str:
    labeled = extract_first_label(text, ("Stop Loss", "Stop-Loss", "止损", "止损价", "风险线"))
    if labeled != "-":
        return labeled
    return extract_sentence(text, ("止损", "风险线", "跌破", "stop loss", "risk line")) or "-"


def extract_sentence(text: str, needles: Iterable[str]) -> str:
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text.replace("\n", " "))
    chunks = re.split(r"(?<=[。.!?；;])\s*", normalized)
    for chunk in chunks:
        if any(needle.lower() in chunk.lower() for needle in needles):
            return clean_value(chunk)
    return ""


def clean_value(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    return value.strip("*` ") or "-"


def count_agent_turns(path: Path, prefix: str) -> int:
    return read_text(path).count(prefix)


def count_scorecard_markers(text: str) -> int:
    return len(re.findall(r"(scorecard|记分牌|优势归属|核心争议|胜出|concede|concession|让步)", text, re.I))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def format_exception(exc: Exception) -> str:
    detail = f"{type(exc).__name__}: {exc}"
    if len(detail) <= RUNNER_ERROR_DETAIL_LIMIT:
        return detail
    return f"{detail[:RUNNER_ERROR_DETAIL_LIMIT].rstrip()}\n...[truncated]"


def safe_print(message: str) -> None:
    try:
        print(message, flush=True)
    except BrokenPipeError:
        pass


def compact_date(date: str) -> str:
    return date.replace("-", "")


if __name__ == "__main__":
    sys.exit(main())
