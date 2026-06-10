"""Local Codex CLI client for TradingAgents.

This provider routes LangChain chat calls through the user's installed
``codex exec`` command instead of an OpenAI-compatible HTTP API. It is meant
for personal/local runs where the user already has Codex CLI authenticated.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from .base_client import BaseLLMClient


_DEFAULT_MODEL_ALIASES = {"", "default", "codex", "codex-cli"}
_CODEX_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}
_TRADING_ANALYSIS_DEPTH_INSTRUCTION = """TradingAgents output-depth contract:
- Preserve the original agent workflow's depth. Do not compress analyst reports, researcher debates, risk debates, or manager decisions into terse summaries.
- When writing a final answer, produce a full investment-research artifact: concrete numbers, dates, price levels, tool evidence, bull and bear counterpoints, risk controls, and the reasoning chain that connects evidence to the recommendation.
- Debate agents must actively engage prior arguments instead of restating a short position. They should preserve old API-style cross-examination: issue-by-issue rebuttals, explicit concessions, evidence tables or scorecards when useful, and direct answers to the opponent's strongest claims. Decision agents must explain why the winning side won and which opposing points still matter.
- Preserve the five-tier portfolio scale. Buy and Sell are high-conviction endpoints; Overweight and Underweight are valid nuanced portfolio ratings for partial exposure changes. Do not collapse "trim", "reduce risk", "avoid adding", or "keep a tracking position" into a full Sell unless the memo actually recommends exiting or avoiding the position outright.
- Keep tactical transaction proposals distinct from portfolio ratings, but do not let that distinction become a bullish upgrade rule. A TraderProposal may use Buy/Hold/Sell for execution, while the Portfolio Manager uses Buy/Overweight/Hold/Underweight/Sell for target exposure.
- Preserve the upstream burden of proof. If the Research Manager is Hold and the TraderProposal is Hold, the Portfolio Manager must normally remain Hold unless the risk debate introduces new, explicit evidence for above-benchmark exposure. Do not upgrade Hold/Hold into Overweight/Buy just because a long-term bull case exists.
- For TraderProposal, Buy means initiating or building a long exposure program, including staged, conditional, or pullback-based entries, but only when the Research Manager recommends Buy/Overweight or explicitly instructs the trader to build, add, restore, or increase exposure. Use Hold when the plan calls for maintaining exposure, waiting for a better entry, awaiting earnings/validation, or monitoring trigger levels without a current add program.
- Weigh evidence according to the requested investment horizon. Do not let a one-day technical move or valuation concern mechanically override already observed fundamentals, backlog/deferred revenue, margin progress, insider/customer evidence, or catalysts; explain why those positives are or are not sufficient.
- In growth-equity debates, separate "not enough safety margin for Buy" from "thesis broken enough for Underweight/Sell". Valuation, negative FCF, leverage, heavy CapEx, or technical weakness can cap sizing and conviction, but should only drive Underweight/Sell after weighing whether realized growth, margin expansion, backlog/deferred revenue, pricing power, customer adoption, catalysts, horizon, and position controls preserve a constructive risk/reward.
- Preserve the actionable horizon from the prompt and upstream agents. Do not stretch a 1-3 month or 3-6 month Hold into a 12-24 month Overweight. When both horizons matter, explicitly separate "short-term/tactical" from "long-term/strategic" inside the narrative; if they conflict, the final rating and TraderProposal should follow the current actionable setup, while the long-term view may be a watchlist or conditional add plan.
- Use the 12-24 month horizon only when the prompt or upstream evidence supports a strategic investment thesis. For a short-term or balanced setup, keep the shorter time horizon and state what long-term evidence would be needed to upgrade.
- Trigger the staged Overweight execution template only after the final rating is actually Overweight or Buy and the upstream evidence supports increasing exposure. The template is appropriate for an evidence-backed above-benchmark program: satellite/growth allocation around 2-4% of the portfolio, staged entries instead of a full first order, an initial tranche around 40-50% of the target position when the current support zone is acceptable, additional tranches near the next support zone or the 50-day SMA, and a hard invalidation/stop at the 50-day SMA unless the prompt provides a different risk budget. When the evidence contains a pullback zone such as 200-210 and a 50-day SMA, include both explicitly: add near 200-210 or closer to the 50-day SMA, and use the 50-day SMA as the hard stop/invalidation line. Do not apply this template to Hold decisions; for Hold, list trigger prices and validation events without calling them an active Buy program.
- Treat MRVL/NXPI-style negative golden cases as guardrails: when the old API-style baseline is Research Manager Hold, Trader Hold, and Portfolio Manager Hold because valuation, timing, or validation risks offset the bull case, Codex must not convert that into Overweight/Buy. In those cases, output a clear short-term Hold/no-add conclusion plus a separate long-term watchlist or conditional-upgrade path.
- For Overweight growth-equity upside targets, preserve the full bull-case target range from the evidence and debate. If the sources or debate include a 280-300 target band, do not collapse the final price target to the lower sell-side anchor merely because one source says 280; use the higher extension target, such as 300, when the final thesis is constructive but position sizing is disciplined.
- Preserve the original debate-coverage depth. Bull, bear, risk, trader, and manager outputs should explicitly address, when source evidence is present: strategic insider ownership such as Leopold Aschenbrenner's stake, GPU pricing power and demand elasticity, Q2/Q3 earnings validation, debt maturity/refinancing structure, deferred-revenue contract quality including refund or take-or-pay terms, steady-state CapEx and GPU refresh-cycle risk, cash versus debt, revenue growth, gross margin, valuation multiples, short interest, and catalysts. If a datapoint is missing, name it as an information gap rather than omitting the topic. Do not turn these topics into a flat checklist; show the clash between the bull and bear interpretation of each material topic.
- Use the requested output language from the conversation. Keep JSON protocols valid when JSON is required, but put the full report text inside the JSON string fields."""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    stripped = _strip_markdown_fence(text)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None

    return payload if isinstance(payload, dict) else None


def _schema_name(schema: Any) -> str:
    return getattr(schema, "__name__", schema.__class__.__name__)


def _schema_json(schema: Any) -> str:
    if hasattr(schema, "model_json_schema"):
        payload = schema.model_json_schema()
    elif hasattr(schema, "schema"):
        payload = schema.schema()
    else:
        payload = {"type": "object"}
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def _validate_schema_payload(schema: Any, payload: dict[str, Any]) -> Any:
    if hasattr(schema, "model_validate"):
        return schema.model_validate(payload)
    return schema(**payload)


def _apply_stop(text: str, stop: list[str] | None) -> str:
    if not stop:
        return text
    earliest = min((idx for s in stop if (idx := text.find(s)) != -1), default=-1)
    return text if earliest == -1 else text[:earliest]


def _tool_name(tool_schema: dict[str, Any]) -> str | None:
    function = tool_schema.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        return str(name) if name else None
    return None


def _normalise_tool_call(raw_call: Any, allowed_names: set[str]) -> dict[str, Any] | None:
    if not isinstance(raw_call, dict):
        return None

    name = raw_call.get("name")
    args = raw_call.get("args")

    function = raw_call.get("function")
    if isinstance(function, dict):
        name = name or function.get("name")
        args = args if args is not None else function.get("arguments")

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}

    if not isinstance(name, str) or name not in allowed_names:
        return None
    if not isinstance(args, dict):
        args = {}

    return {
        "name": name,
        "args": args,
        "id": str(raw_call.get("id") or f"call_{uuid.uuid4().hex}"),
    }


class CodexChatModel(BaseChatModel):
    """LangChain chat model backed by ``codex exec``."""

    model: str = "default"
    command: str = "codex"
    timeout: float = 900.0
    sandbox: str = "read-only"
    approval_policy: str = "never"
    working_dir: Optional[str] = None
    profile: Optional[str] = None
    extra_args: tuple[str, ...] = ()
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    tools: tuple[dict[str, Any], ...] = ()

    @property
    def _llm_type(self) -> str:
        return "codex-cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "command": self.command,
            "sandbox": self.sandbox,
            "approval_policy": self.approval_policy,
            "profile": self.profile,
            "reasoning_effort": self.reasoning_effort,
            "tools": [_tool_name(tool) for tool in self.tools],
        }

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001 - LangChain signature
        tool_schemas = tuple(convert_to_openai_tool(tool) for tool in tools)
        return self._copy_with(tools=tool_schemas)

    def with_structured_output(self, schema, **kwargs):  # noqa: ANN001
        return CodexStructuredChatModel(self, schema)

    def _copy_with(self, **updates: Any) -> "CodexChatModel":
        if hasattr(self, "model_copy"):
            return self.model_copy(update=updates)
        return self.copy(update=updates)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,  # noqa: ANN001
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._build_prompt(messages)
        raw = _apply_stop(self._run_codex(prompt), stop)
        message = self._parse_response(raw)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _build_prompt(self, messages: list[BaseMessage]) -> str:
        sections = [
            "You are the LLM backend inside TradingAgents.",
            "Do not edit files. Do not run repository inspection commands unless the user explicitly asks for codebase work.",
            "Use only the conversation and tool results supplied below for the trading analysis.",
            _TRADING_ANALYSIS_DEPTH_INSTRUCTION,
        ]

        if self.tools:
            sections.append(self._tool_instruction())

        sections.append("Conversation:")
        sections.extend(self._format_message(message) for message in messages)
        return "\n\n".join(section for section in sections if section)

    def _tool_instruction(self) -> str:
        tools_json = json.dumps(list(self.tools), indent=2, ensure_ascii=False)
        return f"""Tools are available, but you cannot execute them yourself.

Available tools:
{tools_json}

If you need tool data, respond with ONLY valid JSON in this shape:
{{"tool_calls":[{{"name":"tool_name","args":{{"arg":"value"}}}}]}}

If the supplied conversation and tool results are enough for the final answer,
respond with ONLY valid JSON in this shape:
{{"content":"final answer text"}}

The content string must contain the complete final report requested by the
agent prompt, not a compressed summary.

Do not wrap the JSON in Markdown."""

    def _format_message(self, message: BaseMessage) -> str:
        role = getattr(message, "type", message.__class__.__name__)
        content = _content_to_text(message.content)

        if role == "tool":
            tool_name = getattr(message, "name", None)
            tool_call_id = getattr(message, "tool_call_id", None)
            label = tool_name or tool_call_id or "tool"
            return f"TOOL RESULT [{label}]:\n{content}"

        if role == "ai" and getattr(message, "tool_calls", None):
            tool_calls = json.dumps(message.tool_calls, ensure_ascii=False, default=str)
            if content:
                return f"ASSISTANT:\n{content}\n\nASSISTANT TOOL CALLS:\n{tool_calls}"
            return f"ASSISTANT TOOL CALLS:\n{tool_calls}"

        return f"{role.upper()}:\n{content}"

    def _run_codex(self, prompt: str) -> str:
        with tempfile.TemporaryDirectory(prefix="tradingagents-codex-") as tmpdir:
            output_path = Path(tmpdir) / "last_message.txt"
            command = self._command(output_path)

            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )

            if completed.returncode != 0:
                stderr = completed.stderr.strip()
                stdout = completed.stdout.strip()
                detail = stderr or stdout or f"exit code {completed.returncode}"
                raise RuntimeError(f"codex exec failed: {detail}")

            if output_path.exists():
                return output_path.read_text(encoding="utf-8").strip()
            return completed.stdout.strip()

    def _command(self, output_path: Path) -> list[str]:
        command = [
            self.command,
            "exec",
            "-",
            "--ephemeral",
            "--color",
            "never",
            "--sandbox",
            self.sandbox,
            "-c",
            f'approval_policy="{self.approval_policy}"',
            "-o",
            str(output_path),
        ]

        workdir = self.working_dir or os.getcwd()
        if workdir:
            command.extend(["-C", workdir])

        if self.profile:
            command.extend(["-p", self.profile])

        model_lower = self.model.lower() if self.model else ""
        reasoning_effort = self.reasoning_effort
        if not reasoning_effort and model_lower in _CODEX_REASONING_EFFORTS:
            reasoning_effort = model_lower

        if reasoning_effort:
            command.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])

        if self.model and model_lower not in _DEFAULT_MODEL_ALIASES | _CODEX_REASONING_EFFORTS:
            command.extend(["-m", self.model])

        command.extend(self.extra_args)
        return command

    def _parse_response(self, raw: str) -> AIMessage:
        if not self.tools:
            return AIMessage(content=raw)

        payload = _extract_json_payload(raw)
        if payload is None:
            return AIMessage(content=raw)

        raw_tool_calls = payload.get("tool_calls")
        allowed_names = {
            name for tool in self.tools if (name := _tool_name(tool)) is not None
        }
        if isinstance(raw_tool_calls, list):
            tool_calls = [
                call
                for raw_call in raw_tool_calls
                if (call := _normalise_tool_call(raw_call, allowed_names)) is not None
            ]
            if tool_calls:
                return AIMessage(content=str(payload.get("content") or ""), tool_calls=tool_calls)

        content = payload.get("content")
        if isinstance(content, str):
            return AIMessage(content=content)
        return AIMessage(content=raw)


class CodexClient(BaseLLMClient):
    """Client for a locally authenticated Codex CLI."""

    def get_llm(self) -> Any:
        extra_args = self.kwargs.get("extra_args") or ()
        if isinstance(extra_args, str):
            extra_args = tuple(shlex.split(extra_args))
        else:
            extra_args = tuple(extra_args)

        return CodexChatModel(
            model=self.model,
            command=self.kwargs.get("command") or os.environ.get("CODEX_BINARY", "codex"),
            timeout=float(self.kwargs.get("timeout", os.environ.get("CODEX_TIMEOUT", 900))),
            sandbox=self.kwargs.get("sandbox", os.environ.get("CODEX_SANDBOX", "read-only")),
            approval_policy=self.kwargs.get(
                "approval_policy",
                os.environ.get("CODEX_APPROVAL_POLICY", "never"),
            ),
            working_dir=self.kwargs.get("working_dir"),
            profile=self.kwargs.get("profile") or os.environ.get("CODEX_PROFILE"),
            extra_args=extra_args,
            temperature=self.kwargs.get("temperature"),
            reasoning_effort=self.kwargs.get("reasoning_effort")
            or os.environ.get("CODEX_REASONING_EFFORT"),
            callbacks=self.kwargs.get("callbacks"),
        )

    def validate_model(self) -> bool:
        return True


class CodexStructuredChatModel:
    """Small structured-output adapter for Codex CLI backed chat calls."""

    def __init__(self, llm: CodexChatModel, schema: Any):
        self.llm = llm
        self.schema = schema

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        response = self.llm.invoke(
            self._with_schema_instruction(input),
            config=config,
            **kwargs,
        )
        raw = response.content if hasattr(response, "content") else str(response)
        payload = _extract_json_payload(raw)
        if payload is None:
            raise ValueError(
                f"Codex structured output for {_schema_name(self.schema)} "
                "did not contain a JSON object."
            )
        return _validate_schema_payload(self.schema, payload)

    def _with_schema_instruction(self, input: Any) -> Any:
        instruction = (
            f"Return ONLY valid JSON for {_schema_name(self.schema)}. "
            "Do not include Markdown fences, prose, comments, or extra keys. "
            "Use the exact enum values shown in the schema, and include all "
            "required fields.\n\n"
            "Structured output is the final TradingAgents report, not a "
            "metadata extraction step. Fill narrative string fields with "
            "complete, evidence-rich prose. For ResearchPlan rationale and "
            "strategic_actions, explain the bull/bear tradeoff, why the chosen "
            "rating won, which opposing risks remain, and concrete execution "
            "steps. For TraderProposal reasoning, include the key evidence and "
            "execution logic rather than a one-line restatement. For "
            "PortfolioDecision executive_summary and investment_thesis, include "
            "a full portfolio-manager decision memo with specific evidence, "
            "risk levels, sizing, catalysts, invalidation conditions, and time "
            "horizon. Preserve rating-scale semantics: choose Buy/Sell only "
            "for endpoint conviction, and choose Overweight/Underweight when "
            "the memo supports partial exposure changes, disciplined sizing, "
            "or tracking positions. For TraderProposal, choose Buy only when "
            "the upstream plan calls for a staged long-entry or "
            "add-to-exposure program, even if entries are conditional or split "
            "across price levels; Hold is for no planned transaction. For "
            "PortfolioDecision, set the rating from the "
            "target exposure implied by the memo, not just from the "
            "TraderProposal action verb, while preserving the upstream burden "
            "of proof: Research Manager Hold plus TraderProposal Hold should "
            "normally remain PortfolioDecision Hold unless the risk debate "
            "adds explicit new evidence for above-benchmark exposure. Include "
            "both a short-term/tactical view and a long-term/strategic view in "
            "PortfolioDecision narrative fields when both are relevant; if "
            "they conflict, keep the final rating tied to the current "
            "actionable setup and express the long-term case as watchlist or "
            "conditional-upgrade language. Preserve 1-3 month or 3-6 month "
            "horizons when upstream evidence supports a short-term Hold. Use "
            "the 12-24 month horizon and staged Overweight execution template "
            "only when the prompt or upstream evidence supports a strategic "
            "Buy/Overweight add program. MRVL/NXPI-style Hold/Hold baselines "
            "are negative golden examples: do not turn them into Overweight/"
            "Buy merely because a long-term bull case exists. For "
            "PortfolioDecision price_target, preserve the neutral target, "
            "current fair-value anchor, or key decision level even when the "
            "final rating is Hold; do not omit Price Target merely because "
            "there is no active Buy program. Do not shorten "
            "fields merely because the response is JSON."
            "\n\n"
            f"{_TRADING_ANALYSIS_DEPTH_INSTRUCTION}\n\nJSON Schema:\n"
            f"{_schema_json(self.schema)}"
        )
        if isinstance(input, str):
            return f"{input}\n\n{instruction}"
        if isinstance(input, list):
            return [*input, {"role": "system", "content": instruction}]
        return input
