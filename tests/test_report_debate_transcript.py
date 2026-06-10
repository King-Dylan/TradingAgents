from __future__ import annotations

from cli.main import save_report_to_disk


def test_save_report_includes_chronological_research_and_risk_transcripts(tmp_path):
    final_state = {
        "investment_debate_state": {
            "history": "Bull Analyst: opening case\nBear Analyst: direct rebuttal",
            "bull_history": "Bull Analyst: opening case",
            "bear_history": "Bear Analyst: direct rebuttal",
            "judge_decision": "Research manager decision",
        },
        "risk_debate_state": {
            "history": (
                "Aggressive Analyst: upside case\n"
                "Conservative Analyst: risk rebuttal\n"
                "Neutral Analyst: balanced answer"
            ),
            "aggressive_history": "Aggressive Analyst: upside case",
            "conservative_history": "Conservative Analyst: risk rebuttal",
            "neutral_history": "Neutral Analyst: balanced answer",
            "judge_decision": "Portfolio manager decision",
        },
    }

    report_file = save_report_to_disk(final_state, "TEST", tmp_path / "TEST_run")
    report_text = report_file.read_text(encoding="utf-8")

    assert (tmp_path / "TEST_run/2_research/debate.md").read_text(encoding="utf-8") == (
        "Bull Analyst: opening case\nBear Analyst: direct rebuttal"
    )
    assert (tmp_path / "TEST_run/2_research/bull.md").exists()
    assert (tmp_path / "TEST_run/2_research/bear.md").exists()
    assert "### Bull/Bear Debate Transcript" in report_text
    assert "Bull Analyst: opening case\nBear Analyst: direct rebuttal" in report_text
    assert report_text.index("### Bull/Bear Debate Transcript") < report_text.index("### Research Manager")

    assert (tmp_path / "TEST_run/4_risk/debate.md").read_text(encoding="utf-8") == (
        "Aggressive Analyst: upside case\n"
        "Conservative Analyst: risk rebuttal\n"
        "Neutral Analyst: balanced answer"
    )
    assert (tmp_path / "TEST_run/4_risk/aggressive.md").exists()
    assert (tmp_path / "TEST_run/4_risk/conservative.md").exists()
    assert (tmp_path / "TEST_run/4_risk/neutral.md").exists()
    assert "### Risk Debate Transcript" in report_text
    assert "Conservative Analyst: risk rebuttal" in report_text
    assert report_text.index("### Risk Debate Transcript") < report_text.index("## V. Portfolio Manager Decision")
