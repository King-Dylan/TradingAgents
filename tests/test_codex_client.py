from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from langchain_core.tools import tool

from tradingagents.agents.schemas import TraderAction, TraderProposal
from tradingagents.llm_clients.codex_client import CodexChatModel, CodexClient
from tradingagents.llm_clients.factory import create_llm_client


@tool
def lookup_price(symbol: str) -> str:
    """Look up a test price."""
    return f"{symbol}=123"


def _fake_codex_run(monkeypatch, payload: str, seen: dict):
    def fake_run(command, input, text, capture_output, timeout, check):
        seen["command"] = command
        seen["input"] = input
        seen["timeout"] = timeout
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_text(payload, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="noisy stdout", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_codex_chat_model_shells_out_and_reads_last_message(monkeypatch, tmp_path):
    seen = {}
    _fake_codex_run(monkeypatch, "final answer", seen)

    llm = CodexChatModel(model="default", working_dir=str(tmp_path), timeout=12)
    result = llm.invoke("Summarize AAPL.")

    assert result.content == "final answer"
    assert seen["command"][:3] == ["codex", "exec", "-"]
    assert "--ephemeral" in seen["command"]
    assert "--sandbox" in seen["command"]
    assert "read-only" in seen["command"]
    approval_index = seen["command"].index("-c")
    assert seen["command"][approval_index + 1] == 'approval_policy="never"'
    assert "-C" in seen["command"]
    assert str(tmp_path) in seen["command"]
    assert "-m" not in seen["command"]
    assert seen["timeout"] == 12


def test_codex_chat_model_passes_custom_model(monkeypatch, tmp_path):
    seen = {}
    _fake_codex_run(monkeypatch, "ok", seen)

    llm = CodexChatModel(model="gpt-5.5", working_dir=str(tmp_path))
    llm.invoke("hello")

    model_index = seen["command"].index("-m")
    assert seen["command"][model_index + 1] == "gpt-5.5"


def test_codex_chat_model_treats_effort_alias_as_reasoning_effort(monkeypatch, tmp_path):
    seen = {}
    _fake_codex_run(monkeypatch, "ok", seen)

    llm = CodexChatModel(model="xhigh", working_dir=str(tmp_path))
    llm.invoke("hello")

    assert "-m" not in seen["command"]
    assert 'model_reasoning_effort="xhigh"' in seen["command"]


def test_codex_chat_model_passes_explicit_reasoning_effort(monkeypatch, tmp_path):
    seen = {}
    _fake_codex_run(monkeypatch, "ok", seen)

    llm = CodexChatModel(
        model="default",
        reasoning_effort="high",
        working_dir=str(tmp_path),
    )
    llm.invoke("hello")

    assert 'model_reasoning_effort="high"' in seen["command"]


def test_codex_bound_tools_parse_json_tool_calls(monkeypatch, tmp_path):
    payload = json.dumps(
        {
            "tool_calls": [
                {"name": "lookup_price", "args": {"symbol": "AAPL"}},
            ]
        }
    )
    seen = {}
    _fake_codex_run(monkeypatch, payload, seen)

    llm = CodexChatModel(model="default", working_dir=str(tmp_path))
    result = llm.bind_tools([lookup_price]).invoke("Need the latest test price.")

    assert "Available tools" in seen["input"]
    assert result.content == ""
    assert result.tool_calls == [
        {
            "name": "lookup_price",
            "args": {"symbol": "AAPL"},
            "id": result.tool_calls[0]["id"],
            "type": "tool_call",
        }
    ]


def test_codex_bound_tools_parse_json_final_content(monkeypatch, tmp_path):
    seen = {}
    _fake_codex_run(monkeypatch, '{"content":"done"}', seen)

    llm = CodexChatModel(model="default", working_dir=str(tmp_path))
    result = llm.bind_tools([lookup_price]).invoke("Use prior tool result.")

    assert result.content == "done"
    assert result.tool_calls == []


def test_codex_structured_output_returns_pydantic_model(monkeypatch, tmp_path):
    seen = {}
    _fake_codex_run(
        monkeypatch,
        '{"action":"Buy","reasoning":"Strong setup.","entry_price":218.0}',
        seen,
    )

    llm = CodexChatModel(model="default", working_dir=str(tmp_path))
    result = llm.with_structured_output(TraderProposal).invoke("Make a proposal.")

    assert isinstance(result, TraderProposal)
    assert result.action is TraderAction.BUY
    assert result.entry_price == 218.0
    assert "JSON Schema" in seen["input"]
    assert "TraderProposal" in seen["input"]


def test_codex_structured_output_adds_schema_instruction_to_message_lists(
    monkeypatch, tmp_path
):
    seen = {}
    _fake_codex_run(
        monkeypatch,
        "```json\n{\"action\":\"Hold\",\"reasoning\":\"Wait for confirmation.\"}\n```",
        seen,
    )

    llm = CodexChatModel(model="default", working_dir=str(tmp_path))
    result = llm.with_structured_output(TraderProposal).invoke(
        [{"role": "user", "content": "Make a proposal."}]
    )

    assert result.action is TraderAction.HOLD
    assert "HUMAN:\nMake a proposal." in seen["input"]
    assert "Return ONLY valid JSON for TraderProposal" in seen["input"]


def test_codex_client_and_factory_return_codex_chat_model(monkeypatch):
    client = create_llm_client(provider="codex", model="default")
    assert isinstance(client, CodexClient)

    llm = client.get_llm()
    assert isinstance(llm, CodexChatModel)


def test_codex_exec_failure_raises(monkeypatch, tmp_path):
    def fake_run(command, input, text, capture_output, timeout, check):
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    llm = CodexChatModel(model="default", working_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="boom"):
        llm.invoke("hello")
