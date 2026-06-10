from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_runner_module():
    path = Path("scripts/run_codex_regression_cases.py")
    spec = importlib.util.spec_from_file_location("run_codex_regression_cases", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compare_to_baseline_flags_key_field_drift_after_outcome_match():
    runner = _load_runner_module()
    baseline = {
        "manager_rating": "Overweight",
        "trader_action": "Buy",
        "pm_rating": "Overweight",
        "price_target": "300.0",
        "entry_price": "222.5",
        "stop_loss": "172.0",
        "position_sizing": "组合的2-4%，分批建仓：第一档40-50%，回落至200-210或172附近50日SMA加仓",
        "time_horizon": "12-24个月",
        "research_debate_transcript": False,
        "risk_debate_transcript": False,
    }
    current = {
        "manager_rating": "Overweight",
        "trader_action": "Buy",
        "pm_rating": "Overweight",
        "price_target": "265.0",
        "entry_price": "218.0",
        "stop_loss": "174.86",
        "position_sizing": "目标总仓位2%-4%；先建立40%-50%，第二 tranche 放在200-210，50日SMA止损",
        "time_horizon": "12-24个月战略持有；1-3个月分批执行",
        "research_debate_transcript": True,
        "risk_debate_transcript": True,
        "complete_has_research_transcript": True,
        "complete_has_risk_transcript": True,
        "bull_rounds": 5,
        "bear_rounds": 5,
        "aggressive_rounds": 5,
        "conservative_rounds": 5,
        "neutral_rounds": 5,
    }

    comparison = runner.compare_to_baseline(current, baseline, strict=True)

    assert comparison["outcome_match"] is True
    assert comparison["key_field_match"] is False
    assert comparison["key_field_mismatches"] == [
        {
            "field": "price_target",
            "baseline": "300.0",
            "current": "265.0",
            "reason": "numeric drift > 5%",
        }
    ]
    assert runner.status_from_result({"failure_count": 0}, comparison) == "FAIL_KEY_FIELDS"


def test_compare_without_baseline_still_checks_transcripts_and_rounds():
    runner = _load_runner_module()
    current = {
        "manager_rating": "Hold",
        "trader_action": "Hold",
        "pm_rating": "Hold",
        "research_debate_transcript": True,
        "risk_debate_transcript": False,
        "complete_has_research_transcript": True,
        "complete_has_risk_transcript": False,
        "bull_rounds": 5,
        "bear_rounds": 5,
        "aggressive_rounds": 5,
        "conservative_rounds": 4,
        "neutral_rounds": 5,
    }

    comparison = runner.compare_to_baseline(current, {}, strict=False)

    assert comparison["available"] is False
    assert comparison["transcript_ok"] is False
    assert comparison["round_ok"] is False
    assert runner.status_from_result({"failure_count": 0}, comparison) == "FAIL_TRANSCRIPT"


def test_runner_exception_summary_is_bounded():
    runner = _load_runner_module()
    message = runner.format_exception(RuntimeError("x" * 3000))

    assert message.startswith("RuntimeError: ")
    assert message.endswith("...[truncated]")
    assert len(message) < 2100
