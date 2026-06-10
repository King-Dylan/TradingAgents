"""Obsidian export helpers for saved TradingAgents reports."""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from tradingagents.agents.utils.rating import parse_rating


TABLE_COLUMNS = (
    "股票",
    "分析时间",
    "版本",
    "PM评级",
    "交易动作",
    "报告",
    "决策",
    "买入价",
    "阶段买入/仓位",
    "止损/风险线",
    "Target",
    "周期",
)


@dataclass(frozen=True)
class AnalysisRow:
    ticker: str
    analysis_time: str
    variant: str
    pm_rating: str
    trader_action: str
    report_link: str
    decision_link: str
    entry_price: str
    position_sizing: str
    stop_loss: str
    price_target: str
    time_horizon: str

    @property
    def run_id(self) -> str:
        match = re.search(r"TradingAgents/Reports/([^|\\\]]+)", self.report_link)
        if match:
            return match.group(1)
        return f"{self.ticker}_{self.analysis_time}_{self.variant}"

    @property
    def sort_key(self) -> tuple[str, str]:
        return (_sortable_datetime(self.analysis_time), self.run_id)

    def to_markdown(self) -> str:
        values = (
            _stock_link(self.ticker),
            self.analysis_time,
            self.variant,
            self.pm_rating,
            self.trader_action,
            self.report_link,
            self.decision_link,
            self.entry_price,
            _shorten(self.position_sizing),
            _shorten(self.stop_loss),
            self.price_target,
            self.time_horizon,
        )
        return "| " + " | ".join(_escape_cell(value) for value in values) + " |"


@dataclass(frozen=True)
class ObsidianExportResult:
    report_path: Path
    decision_path: Path
    stock_path: Path
    summary_path: Path


def export_report_to_obsidian(
    final_state: Mapping,
    ticker: str,
    source_report_file: Path,
    config: Mapping,
    *,
    now: _dt.datetime | None = None,
) -> ObsidianExportResult | None:
    """Write a saved TradingAgents report into an Obsidian vault.

    The analysis pipeline still owns report generation. This exporter only
    mirrors an already-saved report and maintains Obsidian index pages.
    """
    if not config.get("obsidian_auto_export", False):
        return None

    vault_dir_raw = config.get("obsidian_vault_dir")
    if not vault_dir_raw:
        raise ValueError("obsidian_auto_export is enabled but obsidian_vault_dir is empty")

    now = now or _dt.datetime.now()
    ticker = ticker.upper()
    source_report_file = Path(source_report_file).expanduser().resolve()
    if not source_report_file.exists():
        raise FileNotFoundError(source_report_file)

    vault_dir = Path(str(vault_dir_raw)).expanduser()
    tradingagents_dir = config.get("obsidian_tradingagents_dir") or "TradingAgents"
    root = vault_dir / str(tradingagents_dir).strip("/")
    reports_dir = root / "Reports"
    decisions_dir = root / "Decisions"
    stocks_dir = root / "Stocks"
    for directory in (reports_dir, decisions_dir, stocks_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_id = _safe_page_name(source_report_file.parent.name or f"{ticker}_{now:%Y%m%d_%H%M%S}")
    if not run_id.upper().startswith(f"{ticker}_"):
        run_id = _safe_page_name(f"{ticker}_{run_id}")
    variant = _variant_from_run_id(ticker, run_id)
    complete_report = source_report_file.read_text(encoding="utf-8")
    analysis_time = _analysis_time(run_id, complete_report, now)

    pm_text = _portfolio_decision_text(final_state, source_report_file.parent)
    trader_text = str(final_state.get("trader_investment_plan") or "")
    combined_decision_context = "\n\n".join(part for part in (trader_text, pm_text) if part)

    row = AnalysisRow(
        ticker=ticker,
        analysis_time=analysis_time,
        variant=variant,
        pm_rating=parse_rating(pm_text, default="-") if pm_text else "-",
        trader_action=_extract_action(trader_text, pm_text),
        report_link=f"[[TradingAgents/Reports/{run_id}|完整报告]]",
        decision_link=f"[[TradingAgents/Decisions/{run_id} - PM Decision|PM决策]]",
        entry_price=_extract_first_label(combined_decision_context, ("Entry Price", "Entry", "买入价", "入场价")),
        position_sizing=_extract_position_sizing(combined_decision_context),
        stop_loss=_extract_stop_loss(combined_decision_context),
        price_target=_extract_first_label(pm_text, ("Price Target", "Target Price", "Target", "目标价")),
        time_horizon=_extract_first_label(pm_text, ("Time Horizon", "Holding Period", "周期", "时间框架")),
    )

    report_path = reports_dir / f"{run_id}.md"
    decision_path = decisions_dir / f"{run_id} - PM Decision.md"
    stock_path = stocks_dir / f"{ticker}.md"
    summary_path = root / str(config.get("obsidian_summary_page") or "股票分析总览.md")

    report_path.write_text(
        _frontmatter(
            "tradingagents_report",
            ticker,
            run_id,
            variant,
            analysis_time,
            source_report_file,
        )
        + f"\n# {ticker} TradingAgents Report - {analysis_time}\n\n"
        + "[[../股票分析总览|返回股票分析总览]]\n\n"
        + complete_report,
        encoding="utf-8",
    )
    decision_path.write_text(
        _frontmatter(
            "tradingagents_decision",
            ticker,
            run_id,
            variant,
            analysis_time,
            source_report_file.parent / "5_portfolio" / "decision.md",
        )
        + f"\n# {ticker} PM Decision - {analysis_time}\n\n"
        + "[[../股票分析总览|返回股票分析总览]]\n\n"
        + (pm_text or "-"),
        encoding="utf-8",
    )

    existing_rows = _load_rows(summary_path, stocks_dir)
    rows_by_run_id = {existing.run_id: existing for existing in existing_rows}
    rows_by_run_id[row.run_id] = row
    rows = sorted(rows_by_run_id.values(), key=lambda item: item.sort_key, reverse=True)

    summary_path.write_text(_render_summary(rows, now), encoding="utf-8")
    for stock_ticker in sorted({item.ticker for item in rows}):
        stock_rows = [item for item in rows if item.ticker == stock_ticker]
        (stocks_dir / f"{stock_ticker}.md").write_text(
            _render_stock_page(stock_ticker, stock_rows),
            encoding="utf-8",
        )

    _write_index(root / "Reports.md", "完整报告目录", "Reports", reports_dir.glob("*.md"), now)
    _write_index(root / "Decisions.md", "PM 决策目录", "Decisions", decisions_dir.glob("*.md"), now)
    _write_index(root / "Stocks.md", "股票分页目录", "Stocks", stocks_dir.glob("*.md"), now)

    return ObsidianExportResult(
        report_path=report_path,
        decision_path=decision_path,
        stock_path=stock_path,
        summary_path=summary_path,
    )


def _frontmatter(
    doc_type: str,
    ticker: str,
    run_id: str,
    variant: str,
    analysis_time: str,
    source_path: Path,
) -> str:
    return (
        "---\n"
        f'type: "{_yaml_escape(doc_type)}"\n'
        f'ticker: "{_yaml_escape(ticker)}"\n'
        f'run_id: "{_yaml_escape(run_id)}"\n'
        f'variant: "{_yaml_escape(variant)}"\n'
        f'analysis_datetime: "{_yaml_escape(analysis_time)}"\n'
        f'source_path: "{_yaml_escape(str(source_path))}"\n'
        "---\n"
    )


def _portfolio_decision_text(final_state: Mapping, report_dir: Path) -> str:
    for candidate in (
        final_state.get("final_trade_decision"),
        (final_state.get("risk_debate_state") or {}).get("judge_decision")
        if isinstance(final_state.get("risk_debate_state"), Mapping)
        else None,
    ):
        if candidate:
            return str(candidate)

    decision_file = report_dir / "5_portfolio" / "decision.md"
    if decision_file.exists():
        return decision_file.read_text(encoding="utf-8")
    return ""


def _analysis_time(run_id: str, complete_report: str, now: _dt.datetime) -> str:
    match = re.search(r"(\d{8})_(\d{6})", run_id)
    if match:
        return _dt.datetime.strptime("_".join(match.groups()), "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")

    match = re.search(r"^Generated:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)", complete_report, re.M)
    if match:
        value = match.group(1)
        return value if len(value) == 19 else f"{value}:00"

    return now.strftime("%Y-%m-%d %H:%M:%S")


def _variant_from_run_id(ticker: str, run_id: str) -> str:
    suffix = re.sub(rf"^{re.escape(ticker)}_?", "", run_id, flags=re.I)
    suffix = re.sub(r"_?\d{8}_\d{6}$", "", suffix)
    return suffix or "standard"


def _extract_action(trader_text: str, pm_text: str) -> str:
    action = _extract_first_label(trader_text, ("Action", "交易动作", "交易建议"))
    if action != "-":
        return _canonical_action(action)
    match = re.search(r"FINAL TRANSACTION PROPOSAL:\s*\**\s*(BUY|HOLD|SELL)", trader_text, re.I)
    if match:
        return _canonical_action(match.group(1))
    rating = parse_rating(pm_text, default="-")
    if rating in {"Buy", "Overweight"}:
        return "Buy"
    if rating in {"Sell", "Underweight"}:
        return "Sell"
    if rating == "Hold":
        return "Hold"
    return "-"


def _canonical_action(raw: str) -> str:
    clean = raw.strip().strip("*").split()[0].strip(".,;:").lower()
    return {"buy": "Buy", "hold": "Hold", "sell": "Sell"}.get(clean, raw.strip())


def _extract_first_label(text: str, labels: Sequence[str]) -> str:
    if not text:
        return "-"
    for label in labels:
        pattern = re.compile(
            rf"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?\**\s*{re.escape(label)}\s*\**\s*[:：]\s*(.+?)\s*$",
            re.I | re.M,
        )
        match = pattern.search(text)
        if match:
            value = _clean_value(match.group(1))
            if value:
                return value
    return "-"


def _extract_position_sizing(text: str) -> str:
    labeled = _extract_first_label(text, ("Position Sizing", "Position Size", "仓位", "仓位建议", "阶段买入/仓位"))
    if labeled != "-":
        return labeled
    return _extract_sentence(text, ("仓位", "分批", "加仓", "position", "sizing")) or "-"


def _extract_stop_loss(text: str) -> str:
    labeled = _extract_first_label(text, ("Stop Loss", "Stop-Loss", "止损", "止损价", "风险线"))
    if labeled != "-":
        return labeled
    return _extract_sentence(text, ("止损", "风险线", "跌破", "stop loss", "risk line")) or "-"


def _extract_sentence(text: str, needles: Sequence[str]) -> str:
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text.replace("\n", " "))
    chunks = re.split(r"(?<=[。.!?；;])\s*", normalized)
    for chunk in chunks:
        if any(needle.lower() in chunk.lower() for needle in needles):
            return _clean_value(chunk)
    return ""


def _clean_value(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    value = value.strip("*` ")
    return value or "-"


def _render_summary(rows: Sequence[AnalysisRow], now: _dt.datetime) -> str:
    latest_by_ticker: dict[str, AnalysisRow] = {}
    for row in rows:
        latest_by_ticker.setdefault(row.ticker, row)
    latest_rows = sorted(latest_by_ticker.values(), key=lambda item: item.ticker)
    all_rows = sorted(rows, key=lambda item: item.sort_key, reverse=True)
    return (
        "# 股票分析总览\n\n"
        f"> 自动生成自 TradingAgents 本地报告。最后更新：{now:%Y-%m-%d %H:%M:%S}。\n\n"
        "## 最新结论\n\n"
        + _render_table(latest_rows)
        + "\n\n## 全部分析记录\n\n"
        + _render_table(all_rows)
        + "\n\n## 文件夹\n\n"
        "- [[TradingAgents/Reports|完整报告目录]]\n"
        "- [[TradingAgents/Decisions|PM 决策目录]]\n"
        "- [[TradingAgents/Stocks|股票分页目录]]\n"
    )


def _render_stock_page(ticker: str, rows: Sequence[AnalysisRow]) -> str:
    rows = sorted(rows, key=lambda item: item.sort_key, reverse=True)
    latest = rows[0]
    return (
        f"# {ticker} 分析记录\n\n"
        "[[TradingAgents/股票分析总览|返回股票分析总览]]\n\n"
        "## 最新结论\n\n"
        f"- PM评级：{latest.pm_rating}\n"
        f"- 交易动作：{latest.trader_action}\n"
        f"- 买入价：{latest.entry_price}\n"
        f"- 阶段买入/仓位：{latest.position_sizing}\n"
        f"- 止损/风险线：{latest.stop_loss}\n"
        f"- Target：{latest.price_target}\n"
        f"- 周期：{latest.time_horizon}\n"
        f"- 最新报告：{_unescape_link_label(latest.report_link)}\n"
        f"- 最新决策：{_unescape_link_label(latest.decision_link)}\n\n"
        "## 历史分析\n\n"
        + _render_table(rows)
        + "\n"
    )


def _render_table(rows: Sequence[AnalysisRow]) -> str:
    header = "| " + " | ".join(TABLE_COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |"
    body = "\n".join(row.to_markdown() for row in rows)
    return "\n".join(part for part in (header, separator, body) if part)


def _load_rows(summary_path: Path, stocks_dir: Path) -> list[AnalysisRow]:
    rows: list[AnalysisRow] = []
    paths = [summary_path]
    if stocks_dir.exists():
        paths.extend(sorted(stocks_dir.glob("*.md")))
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = _parse_table_row(line)
            if not row or row.run_id in seen:
                continue
            seen.add(row.run_id)
            rows.append(row)
    return rows


def _parse_table_row(line: str) -> AnalysisRow | None:
    if not line.startswith("| [[TradingAgents/Stocks/"):
        return None
    cells = _split_markdown_row(line)
    if len(cells) != len(TABLE_COLUMNS):
        return None
    ticker_match = re.search(r"Stocks/([^|\\\]]+)", cells[0])
    if not ticker_match:
        return None
    return AnalysisRow(
        ticker=ticker_match.group(1).upper(),
        analysis_time=_unescape_cell(cells[1]),
        variant=_unescape_cell(cells[2]),
        pm_rating=_unescape_cell(cells[3]),
        trader_action=_unescape_cell(cells[4]),
        report_link=_unescape_cell(cells[5]),
        decision_link=_unescape_cell(cells[6]),
        entry_price=_unescape_cell(cells[7]),
        position_sizing=_unescape_cell(cells[8]),
        stop_loss=_unescape_cell(cells[9]),
        price_target=_unescape_cell(cells[10]),
        time_horizon=_unescape_cell(cells[11]),
    )


def _split_markdown_row(line: str) -> list[str]:
    inner = line.strip().strip("|")
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in inner:
        if char == "\\" and not escaped:
            escaped = True
            current.append(char)
            continue
        if char == "|" and not escaped:
            cells.append("".join(current).strip())
            current = []
            continue
        current.append(char)
        escaped = False
    cells.append("".join(current).strip())
    return cells


def _write_index(index_path: Path, title: str, folder: str, files, now: _dt.datetime) -> None:
    entries = []
    for file_path in sorted(files, key=lambda path: path.stem.upper()):
        entries.append(f"- [[TradingAgents/{folder}/{file_path.stem}|{file_path.stem}]]")
    index_path.write_text(
        f"# {title}\n\n"
        "[[TradingAgents/股票分析总览|返回股票分析总览]]\n\n"
        f"> 自动生成：{now:%Y-%m-%d %H:%M:%S}。\n\n"
        + ("\n".join(entries) if entries else "- 暂无记录")
        + "\n",
        encoding="utf-8",
    )


def _stock_link(ticker: str) -> str:
    return f"[[TradingAgents/Stocks/{ticker}|{ticker}]]"


def _escape_cell(value: str) -> str:
    return str(value or "-").replace("\n", " ").replace("|", r"\|")


def _unescape_cell(value: str) -> str:
    return value.replace(r"\|", "|").strip()


def _unescape_link_label(value: str) -> str:
    return value.replace(r"\|", "|")


def _shorten(value: str, max_len: int = 260) -> str:
    value = _clean_value(value)
    if len(value) <= max_len:
        return value
    return value[: max_len - 1].rstrip() + "..."


def _sortable_datetime(value: str) -> str:
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})(?::(\d{2}))?", value)
    if not match:
        return value
    year, month, day, hour, minute, second = match.groups()
    return f"{year}{month}{day}{hour}{minute}{second or '00'}"


def _safe_page_name(value: str) -> str:
    value = re.sub(r"[:/\\?#\[\]]+", "_", value.strip())
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .") or "report"


def _yaml_escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')
