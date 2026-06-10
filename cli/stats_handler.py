import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import AIMessage


class StatsCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks LLM calls, tool calls, and token usage."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when an LLM starts."""
        with self._lock:
            self.llm_calls += 1

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when a chat model starts."""
        with self._lock:
            self.llm_calls += 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract token usage from LLM response."""
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            return

        usage_metadata = None
        if hasattr(generation, "message"):
            message = generation.message
            if isinstance(message, AIMessage) and hasattr(message, "usage_metadata"):
                usage_metadata = message.usage_metadata

        if usage_metadata:
            with self._lock:
                self.tokens_in += usage_metadata.get("input_tokens", 0)
                self.tokens_out += usage_metadata.get("output_tokens", 0)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Increment tool call counter when a tool starts."""
        with self._lock:
            self.tool_calls += 1

    def get_stats(self) -> Dict[str, Any]:
        """Return current statistics."""
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
            }


class ToolAuditCallbackHandler(BaseCallbackHandler):
    """Callback handler that records tool outcomes for data-collection audits."""

    FAILURE_STATUSES = {"error", "exception", "not_configured", "rate_limited"}
    _PATTERNS = (
        ("not_configured", re.compile(r"(api key|not configured|missing.*key|placeholder)", re.I)),
        ("rate_limited", re.compile(r"(rate limit|rate-limited|too many requests|quota|http\\s*429|429\\s*too many)", re.I)),
        ("no_data", re.compile(r"(NO_DATA_AVAILABLE|No .* found|No .* reported|no .* data|returned no rows)", re.I)),
        ("error", re.compile(r"(Traceback|Exception|Error (fetching|retrieving|getting)|failed|timed out|timeout)", re.I)),
    )

    def __init__(self, log_path: Optional[Union[str, Path]] = None, preview_chars: int = 700) -> None:
        super().__init__()
        self.log_path = Path(log_path) if log_path else None
        self.preview_chars = preview_chars
        self._lock = threading.Lock()
        self._active: Dict[str, Dict[str, Any]] = {}
        self._counts: Dict[str, int] = {}
        self._failures: List[Dict[str, Any]] = []

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        run_id = self._run_id(kwargs)
        tool_name = self._tool_name(serialized, kwargs)
        record = {
            "event": "tool_start",
            "timestamp": self._timestamp(),
            "run_id": run_id,
            "tool": tool_name,
            "input": self._preview(input_str),
        }

        with self._lock:
            self._active[run_id] = {
                "tool": tool_name,
                "started_at": time.monotonic(),
                "input": self._preview(input_str),
            }
            self._write_record(record)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        run_id = self._run_id(kwargs)
        output_text = self._output_text(output)

        with self._lock:
            active = self._active.pop(run_id, {})
            tool_name = kwargs.get("name") or active.get("tool") or "unknown_tool"
            status, reason = self._classify(output_text)
            self._counts[status] = self._counts.get(status, 0) + 1

            record = {
                "event": "tool_end",
                "timestamp": self._timestamp(),
                "run_id": run_id,
                "tool": tool_name,
                "status": status,
                "reason": reason,
                "duration_ms": self._duration_ms(active),
                "output_chars": len(output_text),
                "output_preview": self._preview(output_text),
            }
            if status in self.FAILURE_STATUSES:
                self._failures.append(record)
            self._write_record(record)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = self._run_id(kwargs)

        with self._lock:
            active = self._active.pop(run_id, {})
            tool_name = kwargs.get("name") or active.get("tool") or "unknown_tool"
            reason = f"{type(error).__name__}: {error}"
            self._counts["exception"] = self._counts.get("exception", 0) + 1

            record = {
                "event": "tool_error",
                "timestamp": self._timestamp(),
                "run_id": run_id,
                "tool": tool_name,
                "status": "exception",
                "reason": self._preview(reason),
                "duration_ms": self._duration_ms(active),
            }
            self._failures.append(record)
            self._write_record(record)

    def get_summary(self) -> Dict[str, Any]:
        """Return aggregate tool audit information."""
        with self._lock:
            total = sum(self._counts.values())
            return {
                "total": total,
                "by_status": dict(sorted(self._counts.items())),
                "failure_count": len(self._failures),
                "failures": list(self._failures),
                "log_path": str(self.log_path) if self.log_path else None,
            }

    def format_summary(self) -> str:
        summary = self.get_summary()
        parts = [f"{status}={count}" for status, count in summary["by_status"].items()]
        counts = ", ".join(parts) if parts else "no tool calls"
        return (
            f"Tool data audit: total={summary['total']}, "
            f"failures={summary['failure_count']}, {counts}"
        )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _run_id(kwargs: Dict[str, Any]) -> str:
        return str(kwargs.get("run_id") or f"tool-{time.monotonic_ns()}")

    @staticmethod
    def _tool_name(serialized: Dict[str, Any], kwargs: Dict[str, Any]) -> str:
        return (
            kwargs.get("name")
            or serialized.get("name")
            or serialized.get("id")
            or "unknown_tool"
        )

    @staticmethod
    def _duration_ms(active: Dict[str, Any]) -> Optional[int]:
        started_at = active.get("started_at")
        if started_at is None:
            return None
        return round((time.monotonic() - started_at) * 1000)

    def _write_record(self, record: Dict[str, Any]) -> None:
        if not self.log_path:
            return
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _preview(self, value: Any) -> str:
        text = self._output_text(value)
        text = " ".join(text.split())
        if len(text) <= self.preview_chars:
            return text
        return text[: self.preview_chars] + "...[truncated]"

    @staticmethod
    def _output_text(value: Any) -> str:
        if value is None:
            return ""
        content = getattr(value, "content", None)
        if content is not None:
            return str(content)
        return str(value)

    def _classify(self, output_text: str) -> tuple[str, str]:
        text = output_text.strip()
        if not text:
            return "empty", "tool returned empty output"

        for status, pattern in self._PATTERNS:
            match = pattern.search(text)
            if match:
                return status, match.group(0)

        if "N/A: Not a trading day (weekend or holiday)" in text:
            return "ok_with_calendar_gaps", "weekend/holiday rows in requested date window"

        return "ok", ""
