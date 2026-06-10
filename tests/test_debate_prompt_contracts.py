from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_research_and_risk_prompts_require_issue_by_issue_debate():
    prompt_files = [
        ROOT / "tradingagents/agents/researchers/bull_researcher.py",
        ROOT / "tradingagents/agents/researchers/bear_researcher.py",
        ROOT / "tradingagents/agents/risk_mgmt/aggressive_debator.py",
        ROOT / "tradingagents/agents/risk_mgmt/conservative_debator.py",
        ROOT / "tradingagents/agents/risk_mgmt/neutral_debator.py",
        ROOT / "tradingagents/llm_clients/codex_client.py",
    ]

    for prompt_file in prompt_files:
        source = prompt_file.read_text(encoding="utf-8")
        assert "issue-by-issue" in source
        assert "scorecard" in source


def test_risk_prompts_allow_auditable_markdown_formatting():
    risk_files = [
        ROOT / "tradingagents/agents/risk_mgmt/aggressive_debator.py",
        ROOT / "tradingagents/agents/risk_mgmt/conservative_debator.py",
        ROOT / "tradingagents/agents/risk_mgmt/neutral_debator.py",
    ]

    for prompt_file in risk_files:
        source = prompt_file.read_text(encoding="utf-8")
        assert "without any special formatting" not in source
        assert "Markdown headings or tables" in source
        assert "4-6 most material current disputes" in source
        assert "Be complete but bounded" in source
