"""
Textual-based TUI implementation (initial skeleton) for the Voice Agent.

This first iteration provides:
- Basic pipeline state model
- Simple chat message model
- Minimal Textual App with:
  * Status bar (static for now, placeholders for dynamic updates)
  * Chat log panel (scrollable)
  * Input panel for text entry (Tab to focus, Enter to send)
- Integration hooks to connect with existing VoiceAgent asynchronously

Later iterations will expand:
- Real pipeline state updates (listening, STT, LLM, TTS, playback)
- Push-to-talk handling
- Keyboard shortcuts (F1 help, F2 settings, F3 debug)
- Color schemes, animations, error notifications
- Right/left alignment (agent left, user right) with timestamps
"""

from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Literal, Deque
from collections import deque
import logging
from pathlib import Path
import json
import time

# Defensive import (allow project to run without textual installed yet)
try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, Container

    # ScrollView may not exist in some Textual versions; provide graceful fallback.
    try:
        from textual.widgets import Static, Input, ScrollView  # type: ignore
    except ImportError:
        from textual.widgets import Static, Input  # type: ignore

        class ScrollView(Static):  # type: ignore
            """Fallback ScrollView: simple Static placeholder (no scrolling)."""

            pass

    from textual.reactive import reactive
    from textual.message import Message
    from textual import events
except ImportError:  # Fallback stubs so rest of project imports do not fail
    # Textual not available; provide minimal stand-ins so module import doesn't crash.
    App = object  # type: ignore
    ComposeResult = object  # type: ignore

    class Static:  # type: ignore
        def __init__(self, *a, **kw):
            pass

    class Input:  # type: ignore
        class Submitted:  # mimic textual Input.Submitted event name used
            def __init__(self, value: str):
                self.value = value

        def __init__(self, *a, **kw):
            self.value = ""
            self.placeholder = kw.get("placeholder", "")

    class ScrollView:  # type: ignore
        def __init__(self, *a, **kw):
            self.size = type("S", (), {"width": 80})

    class Horizontal:  # type: ignore
        pass

    class Vertical:  # type: ignore
        pass

    class Container:  # type: ignore
        pass

    class events:  # type: ignore
        class Key:
            def __init__(self, key: str = ""):
                self.key = key

    class Message:  # type: ignore
        """Fallback stub for textual.message.Message."""

        def __init__(self, *a, **kw):
            pass

    def reactive(value):  # type: ignore
        return value


# ---------------------------------------------------------------------------
# Pipeline / Component State Models
# ---------------------------------------------------------------------------


class ComponentState(Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PipelineStatus:
    audio_input: ComponentState = ComponentState.IDLE
    stt: ComponentState = ComponentState.IDLE
    llm: ComponentState = ComponentState.IDLE
    tts: ComponentState = ComponentState.IDLE
    audio_output: ComponentState = ComponentState.IDLE
    current_message: Optional[str] = None
    last_update: float = 0.0
    error_message: Optional[str] = None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "audio_input": self.audio_input.value,
            "stt": self.stt.value,
            "llm": self.llm.value,
            "tts": self.tts.value,
            "audio_output": self.audio_output.value,
            "current_message": self.current_message,
            "error_message": self.error_message,
            "last_update": self.last_update,
        }


# ---------------------------------------------------------------------------
# Chat Message Model
# ---------------------------------------------------------------------------


@dataclass
class ChatMessage:
    role: Literal["user", "agent", "system", "tool"]
    content: str
    timestamp: dt.datetime = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc)
    )
    status: Literal["sending", "sent", "processing", "complete", "error"] = "complete"
    id: str = field(
        default_factory=lambda: f"msg-{int(asyncio.get_event_loop().time()*1000)}"
    )

    def render_lines(self, width: int = 80) -> List[str]:
        """
        Produce formatted lines for display.
        - Agent: left aligned
        - User: right aligned
        Colors will be applied in the TUI widget (not embedded here to keep pure data).
        """
        ts_local = self.timestamp.astimezone().strftime("%H:%M:%S")
        if self.role == "agent":
            header = f"[{ts_local}] Agent:"
            body = self.content
            return [header, body]
        elif self.role == "user":
            header = f"User [{ts_local}]"
            body = self.content
            # Right alignment will be handled by the rendering widget
            return [header, body]
        else:
            header = f"[{ts_local}] {self.role.capitalize()}:"
            return [header, self.content]


# ---------------------------------------------------------------------------
# Reactive / Event Messages
# ---------------------------------------------------------------------------


class NewChatMessage(Message):
    def __init__(self, message: ChatMessage):
        self.chat_message = message
        super().__init__()


# ---------------------------------------------------------------------------
# UI Widgets
# ---------------------------------------------------------------------------


class StatusBar(Static):
    """
    Simple status bar showing pipeline component states.
    Later: color coding, icons, animations.
    """

    def __init__(
        self,
        get_status: Callable[[], PipelineStatus],
        enable_animations: bool = False,
        get_audio_level: Optional[Callable[[], float]] = None,
        get_microphone_state: Optional[Callable[[], dict]] = None,
    ):
        super().__init__()
        self._get_status = get_status
        self._tick = 0
        self._enable_animations = enable_animations
        self._get_audio_level = get_audio_level or (lambda: 0.0)
        self._get_microphone_state = get_microphone_state or (lambda: {})

    def on_mount(self) -> None:  # type: ignore
        self.set_interval(0.5, self._advance)

    def _advance(self) -> None:
        self._tick += 1
        self.refresh()

    def render(self) -> str:  # type: ignore
        status = self._get_status()
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        def fmt(label: str, state: ComponentState) -> str:
            color_map = {
                ComponentState.IDLE: "grey50",
                ComponentState.INITIALIZING: "yellow",
                ComponentState.READY: "green",
                ComponentState.ACTIVE: "bright_green",
                ComponentState.ERROR: "red",
                ComponentState.DISABLED: "grey30",
            }
            color = color_map[state]
            rendered_label = label
            if state == ComponentState.ACTIVE and self._enable_animations:
                frame = spinner_frames[self._tick % len(spinner_frames)]
                if self._tick % 2 == 1:
                    color = "green"
                rendered_label = f"{label}{frame}"
            return f"[{color}]{rendered_label}={state.value}[/]"

        # Get microphone state for enhanced input display
        mic_state = self._get_microphone_state()

        # Enhanced audio input display with microphone status
        audio_input_display = self._format_audio_input(status.audio_input, mic_state)

        # Base parts (excluding meter so we can right-align meter)
        base_parts = [
            audio_input_display,
            fmt("🧠stt", status.stt),
            fmt("💭llm", status.llm),
            fmt("🗣️tts", status.tts),
            fmt("🔊out", status.audio_output),
        ]
        if status.current_message:
            base_parts.append(f"[cyan]{status.current_message}[/]")
        if status.error_message:
            base_parts.append(f"[red]ERR:{status.error_message}[/]")

        # Audio level meter
        try:
            level = float(self._get_audio_level())
        except Exception:
            level = 0.0
        level = max(0.0, min(level, 1.0))
        bar_width = 12
        filled = int(level * bar_width + 0.5)
        bar = "█" * filled + "░" * (bar_width - filled)
        meter_text = f"[blue]{bar}[/]{int(level*100):3d}%"

        # Join left side
        left = " | ".join(base_parts)

        # Attempt right justification within available width
        try:
            total_width = getattr(self.size, "width", 0) or 0
        except Exception:
            total_width = 0

        if total_width > 0:
            # Rough visible length estimation (strip simple markup [.*?])
            import re

            visible_left = re.sub(r"\[[^\]]+\]", "", left)
            visible_meter = re.sub(r"\[[^\]]+\]", "", meter_text)
            padding = total_width - len(visible_left) - len(visible_meter) - 1
            if padding < 1:
                padding = 1
            return f"{left}{' ' * padding}{meter_text}"
        else:
            # Fallback (no width known) -> append meter at end
            return f"{left} | {meter_text}"

    def _format_audio_input(self, state: ComponentState, mic_state: dict) -> str:
        """Format audio input display with microphone status indicators."""
        # Determine icon and color based on microphone state
        if mic_state.get("is_muted", False):
            icon = "🔇"
            color = "red"
            status_text = "muted"
        elif mic_state.get("is_paused", False):
            icon = "⏸️"
            color = "yellow"
            status_text = "paused"
        elif mic_state.get("has_error", False):
            icon = "❌"
            color = "red"
            status_text = "error"
        elif state == ComponentState.ACTIVE:
            icon = "🎤"
            color = "bright_green"
            status_text = "listening"
        elif state == ComponentState.READY:
            icon = "🎤"
            color = "green"
            status_text = "ready"
        elif state == ComponentState.DISABLED:
            icon = "🚫"
            color = "grey30"
            status_text = "disabled"
        else:
            icon = "🎤"
            color = "grey50"
            status_text = state.value

        # Add animation for active listening
        if state == ComponentState.ACTIVE and self._enable_animations:
            spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            frame = spinner_frames[self._tick % len(spinner_frames)]
            if self._tick % 2 == 1:
                color = "green"
            return f"[{color}]{icon}{frame}={status_text}[/]"

        return f"[{color}]{icon}={status_text}[/]"


class ChatLog(ScrollView):
    """
    Scrollable chat log. Handles basic left/right alignment by padding.
    Respects UI configuration for timestamps and color scheme.
    """

    def __init__(
        self,
        max_messages: int = 200,
        show_timestamps: bool = True,
        color_scheme: str = "default",
    ):
        super().__init__()
        self.max_messages = max_messages
        self._messages: Deque[ChatMessage] = deque(maxlen=max_messages)
        self._show_timestamps = show_timestamps
        self._color_scheme = color_scheme
        # Filter support
        self._filter_term: Optional[str] = None
        self._filtered_cache: List[ChatMessage] = []
        # Pruning / summarization stats
        self._pruned_user = 0
        self._pruned_agent = 0
        # Search match tracking
        self._match_positions: List[int] = []
        self._current_match_index: int = -1

        # Manual scroll state (number of lines user has scrolled back; 0 = bottom)
        self._line_offset: int = 0
        # When user scrolls up, we freeze auto-scroll until they return to bottom
        self._user_scrolled: bool = False

    def add_message(self, msg: ChatMessage) -> None:
        self._messages.append(msg)
        self._maybe_prune()
        self.refresh(layout=True)
        # Auto-follow tail after the refresh if user has not manually scrolled.
        if not getattr(self, "_user_scrolled", False):
            try:
                if hasattr(self, "scroll_end"):
                    # Defer until after layout so ScrollView knows new virtual size.
                    self.call_after_refresh(lambda: self.scroll_end(animate=False))  # type: ignore
            except Exception:
                pass

    def clear_messages(self) -> None:
        self._messages.clear()
        self._pruned_user = 0
        self._pruned_agent = 0
        self.refresh(layout=True)

    def set_filter(self, term: str) -> None:
        """Apply a case-insensitive substring filter and compute match positions."""
        norm = term.strip()
        if not norm:
            self.clear_filter()
            return
        self._filter_term = norm.lower()
        self._filtered_cache = [
            m
            for m in self._messages
            if self._filter_term in m.content.lower()
            or self._filter_term in (m.render_lines()[0].lower())
        ]
        # Record indices within the filtered list for navigation
        self._match_positions = list(range(len(self._filtered_cache)))
        self._current_match_index = 0 if self._match_positions else -1
        self.refresh(layout=True)

    def clear_filter(self) -> None:
        """Clear active filter."""
        if self._filter_term is not None:
            self._filter_term = None
            self._filtered_cache = []
            self._match_positions = []
            self._current_match_index = -1
            self.refresh(layout=True)

    def render(self) -> str:  # type: ignore
        """
        Full render (no manual virtualization) so native ScrollView scrolling & scrollbar
        can function. Tail auto-follow handled in add_message via scroll_end().
        """
        width = self.size.width if hasattr(self.size, "width") else 80

        source = (
            self._filtered_cache
            if self._filter_term is not None
            else list(self._messages)
        )
        highlight_term = self._filter_term
        current_msg_ref: Optional[ChatMessage] = None
        if self._filter_term is not None and 0 <= self._current_match_index < len(
            self._filtered_cache
        ):
            current_msg_ref = self._filtered_cache[self._current_match_index]

        full: List[str] = []
        for msg in source:
            rendered = msg.render_lines(width)
            processed = rendered
            if (
                not self._show_timestamps
                and processed
                and processed[0].startswith("[")
                and "] " in processed[0]
            ):
                first = processed[0]
                closing = first.find("]")
                if closing != -1:
                    processed = [first[closing + 2 :]] + processed[1:]

            if highlight_term:
                processed = [
                    self._highlight_fragment(line, highlight_term) for line in processed
                ]
                if current_msg_ref is msg and processed:
                    processed[0] = f"[bold reverse]{processed[0]}[/]"

            if msg.role == "agent":
                h = "bold cyan"
                b = "cyan"
                if self._color_scheme == "high_contrast":
                    h = "bold bright_cyan"
                    b = "bright_cyan"
                full.append(f"[{h}]{processed[0]}[/]")
                for line in processed[1:]:
                    full.append(f"[{b}]{line}[/]")
                full.append("")
            elif msg.role == "user":
                h = "bold green"
                b = "green"
                if self._color_scheme == "high_contrast":
                    h = "bold bright_green"
                    b = "bright_green"
                for i, line in enumerate(processed):
                    pad = max(0, width - len(line) - 2)
                    color = h if i == 0 else b
                    full.append(" " * pad + f"[{color}]{line}[/]")
                full.append("")
            else:
                c = "magenta"
                if self._color_scheme == "high_contrast":
                    c = "bright_magenta"
                full.append(f"[{c}]{processed[0]}[/]")
                for line in processed[1:]:
                    full.append(f"[{c}]{line}[/]")
                full.append("")

        if full and full[-1] == "":
            full.pop()

        return "\n".join(full)

    # ----- Explicit scroll helpers (improve compatibility across Textual versions) -----
    def scroll_up(self, amount: int = 10) -> None:  # type: ignore
        """
        Scroll upward by a number of lines.
        Provides a stable API for the App actions regardless of Textual version.
        """
        try:
            if hasattr(self, "scroll_relative"):
                # Newer Textual API
                self.scroll_relative(y=-amount)  # type: ignore
            elif hasattr(self, "scroll_to"):
                # Fallback: compute new y from existing (if exposed)
                self.scroll_relative(y=-amount)  # type: ignore
        except Exception:
            pass

    def scroll_down(self, amount: int = 10) -> None:  # type: ignore
        """Scroll downward by a number of lines."""
        try:
            if hasattr(self, "scroll_relative"):
                self.scroll_relative(y=amount)  # type: ignore
            elif hasattr(self, "scroll_to"):
                self.scroll_relative(y=amount)  # type: ignore
        except Exception:
            pass

    @staticmethod
    def _highlight_fragment(text: str, term: str) -> str:
        """Naive case-insensitive substring highlight using inverse style."""
        lower = text.lower()
        start = 0
        parts: List[str] = []
        while True:
            idx = lower.find(term, start)
            if idx == -1:
                parts.append(text[start:])
                break
            parts.append(text[start:idx])
            parts.append(f"[reverse]{text[idx:idx+len(term)]}[/]")
            start = idx + len(term)
        return "".join(parts)

    def _maybe_prune(self) -> None:
        """
        Prune older messages when near capacity and insert a synthesized summary block.
        Strategy:
          - When deque is full, remove oldest 25% (at least 5) and insert a system summary.
        """
        if len(self._messages) < self.max_messages:
            return
        remove_count = max(5, self.max_messages // 4)
        removed: List[ChatMessage] = []
        for _ in range(remove_count):
            if self._messages:
                removed.append(self._messages.popleft())
        # Update stats
        for m in removed:
            if m.role == "user":
                self._pruned_user += 1
            elif m.role == "agent":
                self._pruned_agent += 1
        summary = ChatMessage(
            role="system",
            content=(
                f"(Pruned {len(removed)} earlier messages: "
                f"user={self._pruned_user}, agent={self._pruned_agent}. "
                f"Use /export later to review full history.)"
            ),
        )
        self._messages.appendleft(summary)


class ErrorBanner(Static):
    """Displays an error banner when pipeline_status.error_message is set."""

    def __init__(self, get_status: Callable[[], PipelineStatus]):
        super().__init__()
        self._get_status = get_status

    def on_mount(self) -> None:  # type: ignore
        self.set_interval(0.75, self.refresh)

    def render(self) -> str:  # type: ignore
        status = self._get_status()
        if status.error_message:
            return f"[bold white on red] ERROR: {status.error_message} [/]"
        return ""


class InputPanel(Horizontal):
    """
    Input panel for text entry (fallback / manual queries).
    """

    class Submitted(Message):
        def __init__(self, text: str):
            self.text = text
            super().__init__()

    def __init__(self):
        super().__init__()
        self.input = Input(placeholder="Type (Enter=send, Esc=cancel, /help) ...")
        self._last_text: str = ""

    def compose(self) -> ComposeResult:  # type: ignore
        yield self.input

    async def on_input_submitted(self, event: Input.Submitted) -> None:  # type: ignore
        text = event.value.strip()
        if text:
            self._last_text = text
            # post_message returns a bool (delivered), it's not awaitable
            self.post_message(self.Submitted(text))
            self.input.value = ""

    async def on_key(self, event: events.Key) -> None:  # type: ignore
        # Local input-level key handling:
        # - Esc clears current line.
        # - Ctrl+Q requests app quit. Ctrl+C is intentionally ignored to avoid interfering with terminal copy.
        if event.key == "escape":
            self.input.value = ""
        elif event.key == "ctrl+q":
            try:
                from textual.app import App as _App  # type: ignore

                app = _App.app  # type: ignore[attr-defined]
                if hasattr(app, "action_quit_app"):
                    asyncio.create_task(app.action_quit_app())  # type: ignore
                elif hasattr(app, "exit"):
                    app.exit()  # type: ignore
            except Exception:
                pass
        elif event.key in ("pageup", "pagedown", "home", "end"):
            # Forward scrolling keys even while input focused (use app instance directly)
            try:
                mapping = {
                    "pageup": "action_scroll_chat_up",
                    "pagedown": "action_scroll_chat_down",
                    "home": "action_scroll_chat_top",
                    "end": "action_scroll_chat_bottom",
                }
                method = mapping.get(event.key)
                if method and hasattr(self.app, method):  # type: ignore
                    asyncio.create_task(getattr(self.app, method)())  # type: ignore
                    # Mark user scrolled so auto-scroll pauses
                    if hasattr(self.app, "chat"):
                        setattr(self.app.chat, "_user_scrolled", True)  # type: ignore
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main TUI Application
# ---------------------------------------------------------------------------


class VoiceAgentTUI(App):
    """
    Core Textual App orchestrating UI elements.
    Expects an external "agent_adapter" providing async hooks:
        - send_user_text(text) -> coroutine
        - subscribe(callback)  (optional future extension)
    """

    CSS = """
    Screen {
        layout: vertical;
    }

    #status_bar {
        height: 3;
        border: solid;
        padding: 0 1;
        background: $surface;
    }

    #input_panel {
        dock: bottom;
        height: 3;
        background: $surface;
    }

    #chat_log {
        height: 1fr;
        overflow: auto;
        border: solid;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("ctrl+q", "action_quit_app", "Quit"),
        ("tab", "focus_input", "Focus Input"),
        ("f1", "toggle_help", "Help"),
        ("f2", "toggle_settings", "Settings"),
        ("f3", "toggle_debug", "Debug"),
        ("f4", "toggle_tool_panel", "Tools"),
        ("f5", "push_to_talk", "Voice"),
        ("ctrl+f", "search_chat", "Search"),
        ("ctrl+n", "search_next", "Next Match"),
        ("ctrl+p", "search_prev", "Prev Match"),
        ("pageup", "scroll_chat_up", "Scroll Up"),
        ("page_down", "scroll_chat_up", "Scroll Up"),  # alt key name fallback
        ("pagedown", "scroll_chat_down", "Scroll Down"),
        ("page_down", "scroll_chat_down", "Scroll Down"),  # alt key name fallback
        ("home", "scroll_chat_top", "Scroll Top"),
        ("end", "scroll_chat_bottom", "Scroll Bottom"),
        ("ctrl+l", "clear_chat", "Clear Chat"),
        ("f6", "toggle_dictation", "Dictation"),
        ("f7", "toggle_timestamps", "Timestamps"),
        ("f8", "cycle_colors", "Color Scheme"),
        ("f9", "toggle_animations", "Animations"),
        ("f10", "export_chat", "Export Chat"),
        ("f11", "export_logs", "Export Logs"),
        ("f12", "toggle_metrics_panel", "Metrics"),
        ("ctrl+shift+l", "clear_logs", "Clear Logs"),
    ]

    def __init__(
        self,
        agent_adapter: "AgentAdapter",
        pipeline_status: PipelineStatus,
        ui_config: Optional[Any] = None,
        message_buffer: int = 200,
    ):
        super().__init__()
        self._agent_adapter = agent_adapter
        self._pipeline_status = pipeline_status
        self._ui_config = ui_config or {}
        self.chat = ChatLog(
            max_messages=getattr(self._ui_config, "max_messages", message_buffer),
            show_timestamps=getattr(self._ui_config, "show_timestamps", True),
            color_scheme=getattr(self._ui_config, "color_scheme", "default"),
        )
        self.status_bar = StatusBar(
            lambda: self._pipeline_status,
            enable_animations=getattr(self._ui_config, "enable_animations", True),
            get_audio_level=lambda: (
                getattr(
                    getattr(self._agent_adapter.voice_agent, "audio_manager", None),
                    "last_level",
                    0.0,
                )
                if getattr(self._ui_config, "enable_audio", False)
                else 0.0
            ),
        )
        try:
            self.status_bar.id = "status_bar"  # type: ignore
        except Exception:
            pass
        self.input_panel = InputPanel()
        # Assign stable IDs so CSS docking works
        try:
            self.input_panel.id = "input_panel"  # type: ignore
        except Exception:
            pass
        try:
            self.chat.id = "chat_log"  # type: ignore
        except Exception:
            pass
        self._bg_tasks: List[asyncio.Task] = []

        # Overlays / panels
        self.help_panel: Optional[Static] = None
        self.settings_panel: Optional[Static] = None
        self.debug_panel: Optional[Static] = None
        self.search_panel: Optional[Static] = None
        self.search_input: Optional[Input] = None
        self.tool_panel: Optional[Static] = None
        self.metrics_panel: Optional[Static] = None
        self.error_banner = ErrorBanner(lambda: self._pipeline_status)

        # Tool message cache (for dedicated panel)
        self._tool_messages: Deque[str] = deque(maxlen=20)

        # Metrics storage (recent performance samples)
        self._metrics: Deque[Dict[str, Any]] = deque(maxlen=50)

        self._log_buffer: Deque[str] = deque(maxlen=200)
        # Install TUI log capture handler only in debug mode (root logger DEBUG enabled)
        try:
            import logging as _logging

            if _logging.getLogger().isEnabledFor(_logging.DEBUG):
                self._install_logging_hook()
        except Exception:
            pass

        # Dictation mode state
        self._dictation_mode_active: bool = False
        self._dictation_paused: bool = (
            False  # New: paused (retain accumulated segments, no capture loop)
        )
        self._dictation_task: Optional[asyncio.Task] = None
        self._dictation_segments: List[str] = []
        self._dictation_preview_msg: Optional[ChatMessage] = None
        self._dictation_last_update: float = 0.0

        # Privacy / continuous always-on voice listening
        self._privacy_mode: bool = (
            False  # When True: ignore microphone input & suppress transcriptions
        )
        self._continuous_listen_task: Optional[asyncio.Task] = (
            None  # Background passive listening loop
        )

        # Initialize baseline READY states for text-only minimal components
        self._pipeline_status.llm = ComponentState.READY
        self._pipeline_status.audio_input = ComponentState.DISABLED
        self._pipeline_status.audio_output = ComponentState.DISABLED
        self._pipeline_status.stt = ComponentState.DISABLED
        self._pipeline_status.tts = ComponentState.DISABLED

        # Apply keymap overrides if any
        overrides = getattr(self._ui_config, "keymap_overrides", {}) or {}
        if overrides:
            # Convert {"f5": "clear_chat"} into binding tuples without description (user override)
            for key, action in overrides.items():
                self.BINDINGS.append((key, action, "User"))

    # ---------------------- Composition ----------------------

    def compose(self) -> ComposeResult:  # type: ignore
        yield self.error_banner
        yield self.status_bar

        # Panels (overlay order: help, settings, debug, search)
        self.help_panel = Static(self._help_text(), id="help_panel")
        self.settings_panel = Static(self._settings_text(), id="settings_panel")
        self.debug_panel = Static(self._debug_text(), id="debug_panel")

        # Search panel (initially hidden) with input (Container-based for broader Textual version compatibility)
        self.search_input = Input(
            placeholder="Search (Enter=apply, Esc=close, empty=clear)"
        )
        self.search_panel = Container(self.search_input, id="search_panel")  # type: ignore

        # Tool results panel (initially hidden)
        self.tool_panel = Static(self._tool_panel_text(), id="tool_panel")

        # Metrics panel
        self.metrics_panel = Static(self._metrics_panel_text(), id="metrics_panel")

        for panel in (
            self.help_panel,
            self.settings_panel,
            self.debug_panel,
            self.search_panel,
            self.tool_panel,
            self.metrics_panel,
        ):
            if panel:
                panel.display = False  # type: ignore

        yield self.help_panel  # type: ignore
        yield self.settings_panel  # type: ignore
        yield self.debug_panel  # type: ignore
        yield self.search_panel  # type: ignore
        yield self.tool_panel  # type: ignore
        yield self.metrics_panel  # type: ignore
        yield self.chat
        yield self.input_panel

    # ---------------------- Event Handlers -------------------

    async def on_mount(self) -> None:  # type: ignore
        # Start background task to poll agent messages (if queue-based)
        self._bg_tasks.append(asyncio.create_task(self._consume_agent_queue()))

        # Attempt to activate audio pipeline & start continuous background listening
        try:
            await self._agent_adapter.ensure_audio_pipeline()  # type: ignore
            if self._continuous_listen_task is None:
                self._continuous_listen_task = asyncio.create_task(
                    self._continuous_listen_loop()
                )
        except Exception as e:
            self.chat.add_message(
                ChatMessage(
                    role="system",
                    content=f"(Failed to start continuous listening: {e})",
                    status="error",
                )
            )

        # Set initial focus to input field for immediate typing
        try:
            await self.set_focus(self.input_panel.input)
        except Exception:
            pass

    async def _consume_agent_queue(self) -> None:
        """
        Poll agent adapter queue for new agent messages.
        """
        while True:
            try:
                msg = await self._agent_adapter.receive_agent_message()
                if msg:
                    self.chat.add_message(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Display error internally
                err_msg = ChatMessage(role="system", content=f"Error receiving: {e}")
                self.chat.add_message(err_msg)
            await asyncio.sleep(0.05)

    async def on_input_panel_submitted(self, event: InputPanel.Submitted) -> None:  # type: ignore
        user_text = event.text
        # Display immediately
        self.chat.add_message(ChatMessage(role="user", content=user_text))
        # Send to agent
        asyncio.create_task(self._agent_adapter.handle_user_text(user_text))

    async def action_focus_input(self) -> None:  # type: ignore
        await self.set_focus(self.input_panel.input)

    async def action_clear_chat(self) -> None:  # type: ignore
        self.chat.clear_messages()

    async def action_noop(self) -> None:  # type: ignore
        # Placeholder for future bindings
        pass

    async def on_unmount(self) -> None:  # type: ignore
        for t in self._bg_tasks:
            t.cancel()
        if self._continuous_listen_task:
            self._continuous_listen_task.cancel()
        await asyncio.gather(*self._bg_tasks, return_exceptions=True)

    # ---------------------- Public API -----------------------

    def push_agent_message(self, content: str, status: str = "complete") -> None:
        self.chat.add_message(ChatMessage(role="agent", content=content, status=status))
        if self._is_tool_block(content):
            self._tool_messages.append(content)
            self._refresh_tool_panel()

    # ---------------------- Panel / UI Helpers -----------------------

    def _help_text(self) -> str:
        # Clarify that F5 toggles audio pipeline and performs one-shot capture when active
        voice_active = getattr(self._ui_config, "enable_audio", False)
        voice_line = (
            "F5: Toggle audio pipeline & push-to-talk (one-shot)\n"
            if voice_active
            else "F5: Activate audio pipeline (then push-to-talk)\n"
        )
        return (
            "[bold underline]Help & Shortcuts[/]\n"
            "F1: Toggle this help\n"
            "F2: Settings panel\n"
            "F3: Debug / logs\n"
            "F4: Tool results panel\n"
            f"{voice_line}"
            "F6: Dictation mode start/stop\n"
            "Privacy Voice Commands: 'Privacy Mode' (suspend), 'Privacy Mode Off' (resume)\n"
            "F7: Toggle timestamps\n"
            "F8: Cycle color scheme\n"
            "F9: Toggle animations\n"
            "F10: Export chat\n"
            "F11: Export logs\n"
            "F12: Metrics panel\n"
            "PageUp/PageDown: Scroll chat history; Home: Top; End: Bottom\n"
            "Ctrl+Shift+L: Clear logs\n"
            "Tab: Focus input\n"
            "Ctrl+L: Clear chat\n"
            "Ctrl+F: Search / filter chat\n"
            "Ctrl+N / Ctrl+P: Next / Prev match\n"
            "Ctrl+Q: Quit\n"
            "Esc (in input): Clear input line\n"
        )

    def _settings_text(self) -> str:
        ts = getattr(self._ui_config, "show_timestamps", True)
        cs = getattr(self._ui_config, "color_scheme", "default")
        anim = getattr(self._ui_config, "enable_animations", True)
        dictation = (
            "active" if getattr(self, "_dictation_mode_active", False) else "inactive"
        )
        privacy = "ON" if getattr(self, "_privacy_mode", False) else "off"
        return (
            "[bold underline]Settings[/]\n"
            f"Dictation Mode (F6): {dictation}\n"
            f"Privacy Mode (voice command): {privacy}\n"
            f"Timestamps (F7): {ts}\n"
            f"Color Scheme (F8): {cs}\n"
            f"Animations (F9): {anim}\n"
            "Search (Ctrl+F) - show/hide search panel\n"
            "Pruning: automatic when buffer full (inserts summary)\n"
            "Persistence of changes not yet implemented.\n"
        )

    def _debug_text(self) -> str:
        # Join recent log lines
        lines = list(self._log_buffer)
        tail = "\n".join(lines[-25:])
        return "[bold underline]Debug / Recent Logs[/]\n" + (
            tail or "(no logs captured)"
        )

    def _refresh_debug_panel(self) -> None:
        if self.debug_panel and self.debug_panel.display:  # type: ignore
            self.debug_panel.update(self._debug_text())  # type: ignore

    def _install_logging_hook(self) -> None:
        class _TuiLogHandler(logging.Handler):
            def __init__(self, outer, level=logging.INFO):
                super().__init__(level)
                self.outer = outer

            def emit(self, record: logging.LogRecord) -> None:
                try:
                    msg = self.format(record)
                    self.outer._log_buffer.append(msg)
                    # Defer UI refresh to next loop iteration to avoid cross-thread issues
                    asyncio.get_event_loop().call_soon_threadsafe(
                        self.outer._refresh_debug_panel
                    )
                except Exception:
                    pass

        handler = _TuiLogHandler(self)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(handler)

    # ---------------------- Actions -----------------------

    async def action_toggle_help(self) -> None:  # type: ignore
        if self.help_panel:
            self.help_panel.display = not getattr(self.help_panel, "display")  # type: ignore

    async def action_toggle_settings(self) -> None:  # type: ignore
        if self.settings_panel:
            self.settings_panel.display = not getattr(self.settings_panel, "display")  # type: ignore
            if self.settings_panel.display:  # type: ignore
                self.settings_panel.update(self._settings_text())  # type: ignore

    async def action_toggle_debug(self) -> None:  # type: ignore
        if self.debug_panel:
            self.debug_panel.display = not getattr(self.debug_panel, "display")  # type: ignore
            if self.debug_panel.display:  # type: ignore
                self._refresh_debug_panel()

    async def action_search_chat(self) -> None:  # type: ignore
        # If already visible, hide and clear filter
        if self.search_panel and self.search_panel.display:  # type: ignore
            self.search_panel.display = False  # type: ignore
            self.chat.clear_filter()
            return
        if self.search_panel and self.search_input:
            self.search_panel.display = True  # type: ignore
            await self.set_focus(self.search_input)
            self._update_search_panel_status()

    async def action_toggle_timestamps(self) -> None:  # type: ignore
        current = getattr(self._ui_config, "show_timestamps", True)
        setattr(self._ui_config, "show_timestamps", not current)
        self.chat._show_timestamps = not current
        self.chat.refresh(layout=True)
        if self.settings_panel and self.settings_panel.display:  # type: ignore
            self.settings_panel.update(self._settings_text())  # type: ignore

    async def action_cycle_colors(self) -> None:  # type: ignore
        palette = ["default", "high_contrast"]
        current = getattr(self._ui_config, "color_scheme", "default")
        try:
            idx = palette.index(current)
        except ValueError:
            idx = 0
        new_scheme = palette[(idx + 1) % len(palette)]
        setattr(self._ui_config, "color_scheme", new_scheme)
        self.chat._color_scheme = new_scheme
        self.chat.refresh(layout=True)
        if self.settings_panel and self.settings_panel.display:  # type: ignore
            self.settings_panel.update(self._settings_text())  # type: ignore

    async def action_toggle_animations(self) -> None:  # type: ignore
        current = getattr(self._ui_config, "enable_animations", True)
        setattr(self._ui_config, "enable_animations", not current)
        self.status_bar._enable_animations = not current  # type: ignore
        self.status_bar.refresh()
        if self.settings_panel and self.settings_panel.display:  # type: ignore
            self.settings_panel.update(self._settings_text())  # type: ignore

    async def action_toggle_dictation(self) -> None:  # type: ignore
        """
        Toggle dictation mode.

        On start:
          - Ensure audio pipeline active.
          - Start loop capturing successive audio segments.
          - Transcribe each segment; append to a preview user message (not yet sent to LLM).
        On stop:
          - Finalize aggregated text and send to LLM as a single user turn.
        """
        if not getattr(self, "_dictation_mode_active", False):
            try:
                if not self._agent_adapter.audio_pipeline_active():  # type: ignore
                    activated = await self._agent_adapter.ensure_audio_pipeline()  # type: ignore
                    if activated:
                        self.chat.add_message(
                            ChatMessage(
                                role="system",
                                content="(Audio pipeline activated for dictation)",
                            )
                        )
                self._start_dictation()
                self.chat.add_message(
                    ChatMessage(
                        role="system",
                        content="(Dictation mode started – press F6 to finish)",
                    )
                )
            except Exception as e:
                self.chat.add_message(
                    ChatMessage(
                        role="system",
                        content=f"(Failed to start dictation: {e})",
                        status="error",
                    )
                )
                return
        else:
            await self._stop_dictation(finalize=True)
        if self.settings_panel and self.settings_panel.display:  # type: ignore
            self.settings_panel.update(self._settings_text())  # type: ignore

    async def action_push_to_talk(self) -> None:  # type: ignore
        """
        Audio push-to-talk / toggle:
          - If pipeline inactive: lazily initialize audio/STT/TTS and start one-shot capture.
          - If pipeline active: deactivate (tears down components).

        Current behavior: First press activates & captures. Second press deactivates.
        Disabled while dictation mode is active.
        """
        # Suppress push-to-talk during dictation
        if getattr(self, "_dictation_mode_active", False):
            self.chat.add_message(
                ChatMessage(
                    role="system",
                    content="(Push-to-talk disabled during dictation; press F6 to finish dictation)",
                )
            )
            return
        try:
            if self._agent_adapter.audio_pipeline_active():  # type: ignore
                deactivated = await self._agent_adapter.disable_audio_pipeline()  # type: ignore
                if deactivated:
                    self.chat.add_message(
                        ChatMessage(
                            role="system",
                            content="(Audio pipeline deactivated)",
                        )
                    )
                    if self.help_panel and self.help_panel.display:  # type: ignore
                        self.help_panel.update(self._help_text())  # type: ignore
                    if self.settings_panel and self.settings_panel.display:  # type: ignore
                        self.settings_panel.update(self._settings_text())  # type: ignore
                else:
                    asyncio.create_task(self._agent_adapter.capture_and_process_voice())  # type: ignore
                return

            activated = await self._agent_adapter.ensure_audio_pipeline()  # type: ignore
            if activated:
                self.chat.add_message(
                    ChatMessage(
                        role="system",
                        content="(Audio pipeline activated – press F5 again to deactivate or speak now for capture)",
                    )
                )
                if self.help_panel and self.help_panel.display:  # type: ignore
                    self.help_panel.update(self._help_text())  # type: ignore
                if self.settings_panel and self.settings_panel.display:  # type: ignore
                    self.settings_panel.update(self._settings_text())  # type: ignore
            asyncio.create_task(self._agent_adapter.capture_and_process_voice())  # type: ignore
        except Exception as e:
            self.chat.add_message(
                ChatMessage(
                    role="system",
                    content=f"(Audio toggle/capture failed: {e})",
                    status="error",
                )
            )

    async def action_quit_app(self) -> None:  # type: ignore
        """Explicit quit action to ensure reliable exit (duplicate of default quit)."""
        try:
            self.exit()
        except Exception:
            pass

    async def on_input_submitted(self, event: Input.Submitted) -> None:  # type: ignore
        # Global handler: if the search input is focused, treat submission as filter apply
        if self.search_input and event.input is self.search_input:  # type: ignore[attr-defined]
            term = event.value.strip()
            if term:
                self.chat.set_filter(term)
            else:
                self.chat.clear_filter()
            self._update_search_panel_status()
            return

    async def action_search_next(self) -> None:  # type: ignore
        if self.chat._filter_term and self.chat._match_positions:  # type: ignore
            self.chat._current_match_index = (self.chat._current_match_index + 1) % len(self.chat._match_positions)  # type: ignore
            self.chat.refresh(layout=True)
            self._update_search_panel_status()

    async def action_search_prev(self) -> None:  # type: ignore
        if self.chat._filter_term and self.chat._match_positions:  # type: ignore
            self.chat._current_match_index = (self.chat._current_match_index - 1) % len(self.chat._match_positions)  # type: ignore
            self.chat.refresh(layout=True)
            self._update_search_panel_status()

    async def action_scroll_chat_up(self) -> None:  # type: ignore
        """Scroll up (older content) using native ScrollView."""
        try:
            h = getattr(self.chat.size, "height", 25)
            self.chat._user_scrolled = True  # type: ignore
            amount = max(1, h - 2)
            if hasattr(self.chat, "scroll_relative"):
                self.chat.scroll_relative(y=-amount)  # type: ignore
            elif hasattr(self.chat, "scroll_up"):
                self.chat.scroll_up(amount)  # type: ignore
        except Exception:
            pass

    async def action_scroll_chat_down(self) -> None:  # type: ignore
        """Scroll down (newer content); if near bottom resume auto-follow."""
        try:
            h = getattr(self.chat.size, "height", 25)
            amount = max(1, h - 2)
            if hasattr(self.chat, "scroll_relative"):
                self.chat.scroll_relative(y=amount)  # type: ignore
            elif hasattr(self.chat, "scroll_down"):
                self.chat.scroll_down(amount)  # type: ignore
            # Heuristic: after scrolling down, attempt small scroll_end to detect bottom.
            try:
                # Capture potential prior state by attempting a scroll_end and seeing if it changes.
                if hasattr(self.chat, "scroll_end") and not getattr(
                    self.chat, "_user_scrolled", False
                ):
                    self.chat.scroll_end(animate=False)  # type: ignore
            except Exception:
                pass
        except Exception:
            pass

    async def action_scroll_chat_top(self) -> None:  # type: ignore
        """Jump to very top (oldest)."""
        try:
            self.chat._user_scrolled = True  # type: ignore
            if hasattr(self.chat, "scroll_to"):
                self.chat.scroll_to(y=0)  # type: ignore
            elif hasattr(self.chat, "scroll_relative"):
                # Large negative to ensure top
                self.chat.scroll_relative(y=-1000000)  # type: ignore
        except Exception:
            pass

    async def action_scroll_chat_bottom(self) -> None:  # type: ignore
        """Jump to bottom (newest) & resume auto-follow."""
        try:
            if hasattr(self.chat, "scroll_end"):
                self.chat.scroll_end(animate=False)  # type: ignore
            self.chat._user_scrolled = False  # type: ignore
        except Exception:
            pass

    async def on_key(self, event: events.Key) -> None:  # type: ignore
        """
        Global key handling:
        - Ctrl+Q: quit application.
        - PageUp/PageDown/Home/End: scroll even if input has focus (focus-agnostic scroll).

        Esc no longer quits the app (reserved for clearing input when input focused).
        """
        try:
            if event.key == "ctrl+q":
                await self.action_quit_app()
                return
            if event.key in ("pageup", "pagedown", "home", "end"):
                # Route to scroll actions regardless of current focus
                mapping = {
                    "pageup": self.action_scroll_chat_up,
                    "pagedown": self.action_scroll_chat_down,
                    "home": self.action_scroll_chat_top,
                    "end": self.action_scroll_chat_bottom,
                }
                action = mapping.get(event.key)
                if action:
                    await action()
                return
        except Exception:
            pass

    def _update_search_panel_status(self) -> None:
        """Update search panel placeholder with match navigation status."""
        if not self.search_input:
            return
        try:
            if self.chat._filter_term:  # type: ignore
                total = len(self.chat._filtered_cache)  # type: ignore
                if total:
                    cur = self.chat._current_match_index + 1  # type: ignore
                    self.search_input.placeholder = f"Search '{self.chat._filter_term}' {cur}/{total} (Enter=apply, Esc=close, empty=clear)"  # type: ignore
                else:
                    self.search_input.placeholder = f"Search '{self.chat._filter_term}' 0 matches (Enter=apply, Esc=close, empty=clear)"  # type: ignore
            else:
                self.search_input.placeholder = (
                    "Search (Enter=apply, Esc=close, empty=clear)"
                )
            # Request a refresh so updated placeholder is visible
            self.search_input.refresh()  # type: ignore
        except Exception:
            # Non-fatal; swallow errors if internal representation changes
            pass

    def _tool_panel_text(self) -> str:
        """Render tool panel content."""
        if not self._tool_messages:
            return "[bold underline]Tool Results[/]\n(no tool calls yet)"
        blocks = []
        for raw in list(self._tool_messages)[-8:]:
            blocks.append(raw)
        return "[bold underline]Tool Results (latest)[/]\n" + "\n\n".join(blocks)

    def _metrics_panel_text(self) -> str:
        if not self._metrics:
            return "[bold underline]Performance Metrics[/]\n(no samples yet)"
        lines = [
            "[bold underline]Performance Metrics (latest <= 15)[/]",
            "type | dur_ms | timestamp",
        ]
        for m in list(self._metrics)[-15:]:
            lines.append(f"{m.get('phase')} | {m.get('ms'):>7.1f} | {m.get('ts')}")
        # Aggregate summary last 10 LLM
        llm_samples = [x["ms"] for x in self._metrics if x.get("phase") == "llm"][-10:]
        if llm_samples:
            avg = sum(llm_samples) / len(llm_samples)
            lines.append(f"\nLLM last {len(llm_samples)} avg: {avg:.1f} ms")
        return "\n".join(lines)

    def _refresh_metrics_panel(self) -> None:
        if self.metrics_panel and self.metrics_panel.display:  # type: ignore
            self.metrics_panel.update(self._metrics_panel_text())  # type: ignore

    def _refresh_tool_panel(self) -> None:
        if self.tool_panel and self.tool_panel.display:  # type: ignore
            self.tool_panel.update(self._tool_panel_text())  # type: ignore

    def _record_metric(self, phase: str, ms: float) -> None:
        """
        Record a performance metric sample (used by dictation loop).
        Mirrors adapter behavior but keeps operation local to the TUI instance.
        """
        try:
            ts = dt.datetime.now().strftime("%H:%M:%S")
            self._metrics.append({"phase": phase, "ms": ms, "ts": ts})
            self._refresh_metrics_panel()
        except Exception:
            # Metrics are auxiliary; ignore failures.
            pass

    def _is_tool_block(self, content: str) -> bool:
        return content.startswith("🔧 Tool:")

    async def action_toggle_tool_panel(self) -> None:  # type: ignore
        if self.tool_panel:
            self.tool_panel.display = not getattr(self.tool_panel, "display")  # type: ignore
            if self.tool_panel.display:  # type: ignore
                self.tool_panel.update(self._tool_panel_text())  # type: ignore

    async def action_toggle_metrics_panel(self) -> None:  # type: ignore
        if self.metrics_panel:
            self.metrics_panel.display = not getattr(self.metrics_panel, "display")  # type: ignore
            if self.metrics_panel.display:  # type: ignore
                self._refresh_metrics_panel()

    # ---------------------- Export Utilities -----------------------

    def _export_base_dir(self) -> Path:
        base = Path("exports")
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _collect_chat_messages(self) -> List[ChatMessage]:
        # Include all messages (not only filtered)
        return list(self.chat._messages)  # type: ignore

    def _chat_to_markdown(self, messages: List[ChatMessage]) -> str:
        lines = ["# Chat Export", f"_Exported: {dt.datetime.now().isoformat()}_", ""]
        for m in messages:
            ts = m.timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"### {m.role.capitalize()} ({ts})")
            lines.append("")
            lines.append(m.content.rstrip() or "(empty)")
            lines.append("")
        return "\n".join(lines)

    async def action_export_chat(self) -> None:  # type: ignore
        try:
            msgs = self._collect_chat_messages()
            export_dir = self._export_base_dir()
            ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            md_path = export_dir / f"chat_{ts}.md"
            json_path = export_dir / f"chat_{ts}.json"

            # Write markdown
            md_path.write_text(self._chat_to_markdown(msgs), encoding="utf-8")

            # Write JSON (simple schema)
            json_payload = [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "status": m.status,
                    "id": m.id,
                }
                for m in msgs
            ]
            json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

            self.chat.add_message(
                ChatMessage(
                    role="system",
                    content=f"(Exported chat to {md_path} and {json_path})",
                )
            )
        except Exception as e:
            self.chat.add_message(
                ChatMessage(
                    role="system",
                    content=f"(Chat export failed: {e})",
                    status="error",
                )
            )

    async def action_export_logs(self) -> None:  # type: ignore
        try:
            export_dir = self._export_base_dir()
            ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_path = export_dir / f"logs_{ts}.txt"
            log_path.write_text("\n".join(self._log_buffer), encoding="utf-8")
            self.chat.add_message(
                ChatMessage(
                    role="system",
                    content=f"(Exported logs to {log_path})",
                )
            )
        except Exception as e:
            self.chat.add_message(
                ChatMessage(
                    role="system",
                    content=f"(Log export failed: {e})",
                    status="error",
                )
            )

    async def action_clear_logs(self) -> None:  # type: ignore
        """Clear captured log buffer."""
        self._log_buffer.clear()
        self.chat.add_message(ChatMessage(role="system", content="(Logs cleared)"))
        self._refresh_debug_panel()

    # ---------------------- Voice Command Helpers ----------------------
    @staticmethod
    def detect_voice_command(text: str) -> Optional[str]:
        """
        Enhanced spoken control phrase detector with tolerance for fillers,
        minor STT substitutions, and trailing or leading words.

        Normalized command keys:
          start_dictation, end_dictation, pause_dictation, resume_dictation,
          cancel_dictation, privacy_on, privacy_off
        """
        if not text:
            return None
        norm = text.strip().lower()

        # Strip basic punctuation
        for ch in [".", "!", "?", ","]:
            norm = norm.replace(ch, "")
        # Collapse multiple whitespace
        while "  " in norm:
            norm = norm.replace("  ", " ")

        # Common mis-hearings / substitutions -> canonical
        corrections = {
            "and dictation": "end dictation",
            "in dictation": "end dictation",
            "end the dictation": "end dictation",
            "stop the dictation": "stop dictation",
            "finish the dictation": "finish dictation",
            "send the dictation": "send dictation",
            "pause the dictation": "pause dictation",
            "resume the dictation": "resume dictation",
            "continue the dictation": "continue dictation",
            "cancel the dictation": "cancel dictation",
            "discard the dictation": "discard dictation",
            "abort the dictation": "abort dictation",
        }
        norm = corrections.get(norm, norm)

        def match(
            candidates: List[str], allow_suffix: bool = True, allow_prefix: bool = True
        ) -> bool:
            for c in candidates:
                if norm == c:
                    return True
                if allow_suffix and norm.endswith(c):
                    return True
                if allow_prefix and norm.startswith(c):
                    return True
            return False

        if match(
            [
                "start dictation",
                "take a dictation",
                "begin dictation",
                "start recording",
                "start taking notes",
            ]
        ):
            return "start_dictation"

        if match(
            [
                "end dictation",
                "stop dictation",
                "finish dictation",
                "send dictation",
                "dictation done",
                "end dictation now",
                "stop dictation now",
                "finish dictation now",
            ]
        ):
            return "end_dictation"

        if match(
            [
                "pause dictation",
                "hold dictation",
            ],
            allow_prefix=True,
            allow_suffix=True,
        ):
            return "pause_dictation"

        if match(
            [
                "resume dictation",
                "continue dictation",
                "unpause dictation",
            ]
        ):
            return "resume_dictation"

        if match(
            [
                "cancel dictation",
                "discard dictation",
                "abort dictation",
            ]
        ):
            return "cancel_dictation"

        # Privacy mode detection - order matters and we need exact matching
        # Check "privacy mode off" first since it's more specific
        if match(
            [
                "privacy mode off",
                "resume listening",
            ]
        ):
            return "privacy_off"

        if match(
            [
                "privacy mode",
                "privacy mode on",
                "stop listening",
            ]
        ):
            return "privacy_on"

        if match(
            [
                "end program",
                "quit",
                "exit",
                "shutdown",
                "close application",
                "terminate",
                "quit application",
                "exit application",
                "shut down",
                "close program",
                "terminate program",
            ]
        ):
            return "quit_app"

        return None

    async def handle_voice_command(self, command: str) -> bool:
        """
        Execute a detected voice command.
        Returns True if a command was handled (and default processing should stop).
        """
        try:
            if command == "start_dictation":
                if self._privacy_mode:
                    self.chat.add_message(
                        ChatMessage(
                            role="system",
                            content="(Cannot start dictation while Privacy Mode is ON)",
                        )
                    )
                    return True
                if self._dictation_mode_active and not self._dictation_paused:
                    self.chat.add_message(
                        ChatMessage(role="system", content="(Dictation already active)")
                    )
                    return True
                if self._dictation_mode_active and self._dictation_paused:
                    await self._resume_dictation()
                    return True
                self._start_dictation()
                self.chat.add_message(
                    ChatMessage(
                        role="system",
                        content="(Dictation mode started – speak freely, say 'End dictation' when finished)",
                    )
                )
                if self.settings_panel and self.settings_panel.display:  # type: ignore
                    self.settings_panel.update(self._settings_text())  # type: ignore
                return True
            elif command == "resume_dictation":
                if not self._dictation_mode_active:
                    self.chat.add_message(
                        ChatMessage(
                            role="system", content="(No dictation session to resume)"
                        )
                    )
                    return True
                if not self._dictation_paused:
                    self.chat.add_message(
                        ChatMessage(role="system", content="(Dictation not paused)")
                    )
                    return True
                await self._resume_dictation()
                return True
            elif command == "pause_dictation":
                if not self._dictation_mode_active:
                    self.chat.add_message(
                        ChatMessage(
                            role="system", content="(No active dictation to pause)"
                        )
                    )
                    return True
                if self._dictation_paused:
                    self.chat.add_message(
                        ChatMessage(role="system", content="(Dictation already paused)")
                    )
                    return True
                await self._pause_dictation()
                return True
            elif command == "end_dictation":
                if not self._dictation_mode_active:
                    self.chat.add_message(
                        ChatMessage(
                            role="system", content="(No active dictation to end)"
                        )
                    )
                    return True
                await self._stop_dictation(finalize=True)
                return True
            elif command == "cancel_dictation":
                if not self._dictation_mode_active:
                    self.chat.add_message(
                        ChatMessage(
                            role="system", content="(No active dictation to cancel)"
                        )
                    )
                    return True
                await self._stop_dictation(finalize=False)
                return True
            elif command == "privacy_on":
                if not self._privacy_mode:
                    # Cancel any dictation in progress
                    if self._dictation_mode_active:
                        await self._stop_dictation(finalize=False)
                    self._privacy_mode = True
                    self.chat.add_message(
                        ChatMessage(
                            role="system",
                            content="(Privacy Mode ON – audio ignored until 'Privacy Mode Off')",
                        )
                    )
                    if self.settings_panel and self.settings_panel.display:  # type: ignore
                        self.settings_panel.update(self._settings_text())  # type: ignore
                else:
                    self.chat.add_message(
                        ChatMessage(role="system", content="(Privacy Mode already ON)")
                    )
                return True
            elif command == "privacy_off":
                if self._privacy_mode:
                    self._privacy_mode = False
                    self.chat.add_message(
                        ChatMessage(
                            role="system",
                            content="(Privacy Mode OFF – resuming active listening)",
                        )
                    )
                    if (
                        self._continuous_listen_task is None
                    ) or self._continuous_listen_task.done():
                        self._continuous_listen_task = asyncio.create_task(
                            self._continuous_listen_loop()
                        )
                    if self.settings_panel and self.settings_panel.display:  # type: ignore
                        self.settings_panel.update(self._settings_text())  # type: ignore
                else:
                    self.chat.add_message(
                        ChatMessage(role="system", content="(Privacy Mode already off)")
                    )
                return True
            elif command == "quit_app":
                self.chat.add_message(
                    ChatMessage(
                        role="system",
                        content="(Quitting application via voice command...)",
                    )
                )
                # Call the existing quit action method
                asyncio.create_task(self.action_quit_app())
                return True
        except Exception as e:
            self.chat.add_message(
                ChatMessage(
                    role="system",
                    content=f"(Voice command error: {e})",
                    status="error",
                )
            )
            return True
        return False

    # ---------------------- Dictation Helpers ----------------------
    def _start_dictation(self) -> None:
        if self._dictation_mode_active:
            return
        self._dictation_mode_active = True
        self._dictation_paused = False
        self._dictation_segments = []
        self._dictation_preview_msg = ChatMessage(
            role="user", content="(dictating...) █"
        )
        self.chat.add_message(self._dictation_preview_msg)
        self._dictation_task = asyncio.create_task(self._dictation_loop())

    async def _pause_dictation(self) -> None:
        if not self._dictation_mode_active or self._dictation_paused:
            return
        self._dictation_paused = True
        # Cancel loop
        if self._dictation_task:
            self._dictation_task.cancel()
            try:
                await self._dictation_task
            except Exception:
                pass
            self._dictation_task = None
        # Remove cursor block
        if (
            self._dictation_preview_msg
            and self._dictation_preview_msg.content.endswith(" █")
        ):
            self._dictation_preview_msg.content = self._dictation_preview_msg.content[
                :-2
            ]
            self.chat.refresh(layout=True)
        self.chat.add_message(
            ChatMessage(
                role="system",
                content="(Dictation paused – say 'Resume dictation' to continue, 'End dictation' to finalize, or 'Cancel dictation' to discard)",
            )
        )
        if self.settings_panel and self.settings_panel.display:  # type: ignore
            self.settings_panel.update(self._settings_text())  # type: ignore

    async def _resume_dictation(self) -> None:
        if not self._dictation_mode_active or not self._dictation_paused:
            return
        self._dictation_paused = False
        # Reinstate cursor indicator
        if (
            self._dictation_preview_msg
            and not self._dictation_preview_msg.content.endswith(" █")
        ):
            self._dictation_preview_msg.content = (
                self._dictation_preview_msg.content + " █"
            )
            self.chat.refresh(layout=True)
        self._dictation_task = asyncio.create_task(self._dictation_loop())
        self.chat.add_message(
            ChatMessage(
                role="system",
                content="(Dictation resumed – continue speaking, say 'Pause dictation' to pause again)",
            )
        )
        if self.settings_panel and self.settings_panel.display:  # type: ignore
            self.settings_panel.update(self._settings_text())  # type: ignore

    async def _stop_dictation(self, finalize: bool) -> None:
        """
        Stop dictation mode, finalize (or discard) aggregated text, and
        normalize any lingering ACTIVE pipeline states (audio_input/stt) so
        the status bar does not continue animating post‑dictation.
        """
        if not self._dictation_mode_active:
            return

        self._dictation_mode_active = False

        # Cancel the capture loop task (may currently be inside audio_manager.listen()).
        if self._dictation_task:
            self._dictation_task.cancel()
            try:
                await self._dictation_task
            except Exception:
                pass
            self._dictation_task = None

        # Force-reset pipeline component states that may have been left ACTIVE if
        # cancellation occurred while listen() had emitted "active" but not yet "ready".
        try:
            ps = self._pipeline_status
            if ps:
                if ps.audio_input == ComponentState.ACTIVE:
                    ps.audio_input = ComponentState.READY
                if ps.stt == ComponentState.ACTIVE:
                    ps.stt = ComponentState.READY
                # Clear transient message if it was a dictation-related status.
                if ps.current_message and "listening" in ps.current_message.lower():
                    ps.current_message = None
        except Exception:
            pass

        final_text = " ".join(self._dictation_segments).strip()

        if self._dictation_preview_msg:
            if finalize and final_text:
                # Commit final aggregated content BUT DO NOT send to LLM (per requirement).
                # User transcript remains in chat; no agent response is expected.
                self._dictation_preview_msg.content = final_text
                self.chat.refresh(layout=True)
            else:
                # Treat empty (or canceled) dictation as a system notice.
                if not final_text:
                    self._dictation_preview_msg.role = "system"
                    self._dictation_preview_msg.content = (
                        "(Dictation canceled – no speech captured)"
                    )
                    self.chat.refresh(layout=True)

        self.chat.add_message(
            ChatMessage(
                role="system",
                content=(
                    "(Dictation finished – transcript not sent to agent)"
                    if finalize
                    else "(Dictation canceled)"
                ),
            )
        )

        # Clear preview references for next session.
        self._dictation_preview_msg = None
        self._dictation_segments = []

        # Refresh status bar explicitly so UI immediately reflects normalized states.
        try:
            self.status_bar.refresh()
        except Exception:
            pass

    async def _dictation_loop(self) -> None:
        va = getattr(self._agent_adapter, "voice_agent", None)
        if not va:
            return
        while self._dictation_mode_active:
            try:
                audio = await va.audio_manager.listen()  # type: ignore
                if audio is None:
                    continue
                t_stt = time.monotonic()
                text = await va.stt_service.transcribe(audio)  # type: ignore
                self._record_metric("stt", (time.monotonic() - t_stt) * 1000.0)
                text = text.strip()
                if not text:
                    continue

                # In-dictation voice command handling (pause / resume / end / cancel / privacy)
                cmd = self.detect_voice_command(text)
                if cmd == "pause_dictation":
                    await self._pause_dictation()
                    continue
                if cmd == "end_dictation":
                    await self._stop_dictation(finalize=True)
                    continue
                if cmd == "cancel_dictation":
                    await self._stop_dictation(finalize=False)
                    continue
                if cmd in ("start_dictation", "resume_dictation"):
                    # Redundant inside active (non-paused) session
                    continue
                if cmd in ("privacy_on", "privacy_off"):
                    handled = await self.handle_voice_command(cmd)
                    if self._privacy_mode:
                        # Privacy ON cancels dictation already
                        break
                    if handled:
                        continue

                self._dictation_segments.append(text)
                if self._dictation_preview_msg:
                    joined = " ".join(self._dictation_segments)
                    suffix = " █" if not self._dictation_paused else ""
                    self._dictation_preview_msg.content = joined + suffix
                    self.chat.refresh(layout=True)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.chat.add_message(
                    ChatMessage(
                        role="system",
                        content=f"(Dictation error: {e})",
                        status="error",
                    )
                )
                await asyncio.sleep(0.3)
                continue

    async def _continuous_listen_loop(self) -> None:
        """
        Background continuous listening loop:
          - Runs whenever audio pipeline active & not in dictation
          - Skips processing while privacy mode is ON
          - Performs VAD-gated capture → STT → voice command dispatch
          - Normal utterances go directly to LLM + TTS (hands‑free mode)
        """
        va = getattr(self._agent_adapter, "voice_agent", None)
        if not va:
            return
        while True:
            try:
                # Respect privacy mode
                if self._privacy_mode:
                    await asyncio.sleep(0.3)
                    continue
                # Skip while dictation owns microphone
                if self._dictation_mode_active:
                    await asyncio.sleep(0.1)
                    continue

                audio = await va.audio_manager.listen()  # type: ignore
                if audio is None:
                    continue

                t_stt = time.monotonic()
                text = await va.stt_service.transcribe(audio)  # type: ignore
                self._record_metric("stt", (time.monotonic() - t_stt) * 1000.0)
                text = (text or "").strip()
                if not text:
                    continue

                # Voice command interception (privacy / dictation)
                cmd = self.detect_voice_command(text)
                if cmd:
                    handled = await self.handle_voice_command(cmd)
                    if handled:
                        continue

                # State may have changed after command handling
                if self._privacy_mode or self._dictation_mode_active:
                    continue

                # Normal conversational turn
                self.chat.add_message(ChatMessage(role="user", content=text))
                try:
                    t_llm = time.monotonic()
                    response = await va.llm_service.generate_response(  # type: ignore
                        text, va.conversation_history, va.tool_executor  # type: ignore
                    )
                    self._record_metric("llm", (time.monotonic() - t_llm) * 1000.0)
                    self.chat.add_message(ChatMessage(role="agent", content=response))
                    if getattr(va, "tts_service", None):
                        t_tts = time.monotonic()
                        await va.tts_service.speak(response)  # type: ignore
                        self._record_metric("tts", (time.monotonic() - t_tts) * 1000.0)
                    if hasattr(va, "_update_history"):
                        va._update_history(text, response)  # type: ignore
                except Exception as convo_err:
                    self.chat.add_message(
                        ChatMessage(
                            role="system",
                            content=f"(Conversation error: {convo_err})",
                            status="error",
                        )
                    )
                    await asyncio.sleep(0.25)
            except asyncio.CancelledError:
                break
            except Exception as loop_err:
                self.chat.add_message(
                    ChatMessage(
                        role="system",
                        content=f"(Listen loop error: {loop_err})",
                        status="error",
                    )
                )
                await asyncio.sleep(0.5)
                continue


# ---------------------------------------------------------------------------
# Agent Adapter - Bridges VoiceAgent logic with TUI
# ---------------------------------------------------------------------------


class AgentAdapter:
    """
    Adapter to decouple the TUI from the existing VoiceAgent implementation.

    For this initial version, we assume:
      - VoiceAgent instance is already initialized outside
      - We'll use process_text() for text input path
      - Future: integrate real-time audio events for status updates
    """

    def __init__(self, voice_agent: Any, pipeline_status: PipelineStatus | None = None):
        self.voice_agent = voice_agent
        self._agent_queue: asyncio.Queue[ChatMessage] = asyncio.Queue()
        self._pipeline_status = pipeline_status
        self._voice_task: Optional[asyncio.Task] = None
        self._audio_activation_done = False
        self._audio_deactivated = (
            False  # User toggle flag: True after explicit deactivation
        )

    async def ensure_audio_pipeline(self) -> bool:
        """
        Lazily initialize audio/STT/TTS components if they were skipped due to text_only
        or disabled ui settings. Returns True if activation occurred this call.
        """
        # If previously deactivated, clear flag so we can recreate
        self._audio_deactivated = False
        va = self.voice_agent
        # Already active?
        if all(
            [
                getattr(va, "audio_manager", None),
                getattr(va, "stt_service", None),
                getattr(va, "tts_service", None),
            ]
        ):
            return False

        activated = False
        try:
            if getattr(va, "text_only", False):
                try:
                    va.text_only = False  # type: ignore
                except Exception:
                    pass

            state_cb = getattr(va, "_state_callback", None)

            if not getattr(va, "audio_manager", None):
                from voice_agent.core.audio_manager import AudioManager  # type: ignore

                va.audio_manager = AudioManager(va.config.audio, state_callback=state_cb)  # type: ignore
                await va.audio_manager.initialize()  # type: ignore
                activated = True
                if self._pipeline_status:
                    self._pipeline_status.audio_input = ComponentState.READY
                    self._pipeline_status.audio_output = ComponentState.READY

            if not getattr(va, "stt_service", None):
                from voice_agent.core.stt_service import STTService  # type: ignore

                va.stt_service = STTService(va.config.stt, state_callback=state_cb)  # type: ignore
                await va.stt_service.initialize()  # type: ignore
                activated = True
                if self._pipeline_status:
                    self._pipeline_status.stt = ComponentState.READY

            if not getattr(va, "tts_service", None):
                from voice_agent.core.tts_service import TTSService  # type: ignore

                va.tts_service = TTSService(va.config.tts, va.audio_manager, state_callback=state_cb)  # type: ignore
                await va.tts_service.initialize()  # type: ignore
                activated = True
                if self._pipeline_status:
                    self._pipeline_status.tts = ComponentState.READY

            ui_cfg = getattr(getattr(va, "config", None), "ui", None)
            if ui_cfg and hasattr(ui_cfg, "enable_audio"):
                try:
                    setattr(ui_cfg, "enable_audio", True)
                except Exception:
                    pass

            self._audio_activation_done = self._audio_activation_done or activated
            return activated
        except Exception as e:
            if activated and self._pipeline_status:
                self._pipeline_status.error_message = f"audio activation failed: {e}"
            raise

    def audio_pipeline_active(self) -> bool:
        va = self.voice_agent
        return not self._audio_deactivated and all(
            [
                getattr(va, "audio_manager", None),
                getattr(va, "stt_service", None),
                getattr(va, "tts_service", None),
            ]
        )

    async def disable_audio_pipeline(self) -> bool:
        """
        Deactivate audio pipeline: cleanup services and mark components disabled.
        Returns True if deactivation occurred.
        """
        va = self.voice_agent
        if not self.audio_pipeline_active():
            return False
        deactivated = False
        try:
            if getattr(va, "tts_service", None):
                try:
                    await va.tts_service.cleanup()  # type: ignore
                except Exception:
                    pass
                va.tts_service = None  # type: ignore
                deactivated = True
            if getattr(va, "stt_service", None):
                try:
                    await va.stt_service.cleanup()  # type: ignore
                except Exception:
                    pass
                va.stt_service = None  # type: ignore
                deactivated = True
            if getattr(va, "audio_manager", None):
                try:
                    await va.audio_manager.cleanup()  # type: ignore
                except Exception:
                    pass
                va.audio_manager = None  # type: ignore
                deactivated = True

            if self._pipeline_status:
                self._pipeline_status.audio_input = ComponentState.DISABLED
                self._pipeline_status.audio_output = ComponentState.DISABLED
                self._pipeline_status.stt = ComponentState.DISABLED
                self._pipeline_status.tts = ComponentState.DISABLED

            ui_cfg = getattr(getattr(va, "config", None), "ui", None)
            if ui_cfg and hasattr(ui_cfg, "enable_audio"):
                try:
                    setattr(ui_cfg, "enable_audio", False)
                except Exception:
                    pass

            self._audio_deactivated = True
            return deactivated
        except Exception:
            self._audio_deactivated = True
            return deactivated

    async def handle_user_text(self, text: str) -> None:
        try:
            if self._pipeline_status:
                self._pipeline_status.llm = ComponentState.ACTIVE
                self._pipeline_status.current_message = "Generating response..."
            response = await self.voice_agent.process_text(text)
            await self._agent_queue.put(ChatMessage(role="agent", content=response))
        except Exception as e:
            await self._agent_queue.put(
                ChatMessage(role="system", content=f"Error: {e}", status="error")
            )
        finally:
            if self._pipeline_status:
                self._pipeline_status.llm = ComponentState.READY
                self._pipeline_status.current_message = None

    async def receive_agent_message(self) -> Optional[ChatMessage]:
        try:
            msg = await self._agent_queue.get()
            return msg
        except Exception:
            return None

    async def capture_and_process_voice(self) -> None:
        """One-shot push-to-talk capture + STT + LLM + TTS pipeline."""
        if self._voice_task and not self._voice_task.done():
            # Already capturing
            return
        self._voice_task = asyncio.create_task(self._do_capture())

    async def _do_capture(self) -> None:
        va = self.voice_agent
        try:
            if not all(
                [
                    getattr(va, "audio_manager", None),
                    getattr(va, "stt_service", None),
                    getattr(va, "tts_service", None),
                    getattr(va, "llm_service", None),
                    getattr(va, "tool_executor", None),
                ]
            ):
                await self._agent_queue.put(
                    ChatMessage(
                        role="system",
                        content="(Voice pipeline not available - enable_audio=True & components must be initialized)",
                    )
                )
                return

            await self._agent_queue.put(
                ChatMessage(role="system", content="🎙️ Listening...")
            )
            audio = await va.audio_manager.listen()  # type: ignore
            if audio is None:
                await self._agent_queue.put(
                    ChatMessage(role="system", content="(No speech detected)")
                )
                return

            t_stt = time.monotonic()
            text = await va.stt_service.transcribe(audio)  # type: ignore
            self._record_metric("stt", (time.monotonic() - t_stt) * 1000.0)
            text = text.strip()
            if not text:
                await self._agent_queue.put(
                    ChatMessage(role="system", content="(Empty transcription)")
                )
                return

            # Attempt voice command interception (outside dictation mode)
            try:
                from textual.app import App as _App  # type: ignore

                tui = _App.app  # type: ignore[attr-defined]
            except Exception:
                tui = None
            if tui and hasattr(tui, "detect_voice_command"):
                try:
                    cmd = tui.detect_voice_command(text)  # type: ignore
                except Exception:
                    cmd = None
                if cmd:
                    try:
                        handled = await tui.handle_voice_command(cmd)  # type: ignore
                    except Exception:
                        handled = False
                    if handled:
                        return  # Command consumed (e.g., started/ended dictation)

            await self._agent_queue.put(ChatMessage(role="user", content=text))

            t_llm = time.monotonic()
            response = await va.llm_service.generate_response(  # type: ignore
                text, va.conversation_history, va.tool_executor  # type: ignore
            )
            self._record_metric("llm", (time.monotonic() - t_llm) * 1000.0)

            await self._agent_queue.put(ChatMessage(role="agent", content=response))

            t_tts = time.monotonic()
            await va.tts_service.speak(response)  # type: ignore
            self._record_metric("tts", (time.monotonic() - t_tts) * 1000.0)

            if hasattr(va, "_update_history"):
                va._update_history(text, response)  # type: ignore
        except Exception as e:
            await self._agent_queue.put(
                ChatMessage(role="system", content=f"Voice error: {e}", status="error")
            )

    def _record_metric(self, phase: str, ms: float) -> None:
        """
        Record a performance metric sample into the active TUI App (if present).
        Uses best-effort access to the running Textual App singleton.
        """
        try:
            ts = dt.datetime.now().strftime("%H:%M:%S")
            from textual.app import App as _App  # type: ignore

            tui = _App.app  # type: ignore[attr-defined]
            if hasattr(tui, "_metrics"):  # type: ignore
                tui._metrics.append({"phase": phase, "ms": ms, "ts": ts})  # type: ignore
                if hasattr(tui, "_refresh_metrics_panel"):  # type: ignore
                    tui._refresh_metrics_panel()  # type: ignore
        except Exception:
            # Silent; metrics are auxiliary
            pass


# ---------------------------------------------------------------------------
# Helper to launch TUI
# ---------------------------------------------------------------------------


async def run_tui(
    voice_agent: Any, pipeline_status: Optional[PipelineStatus] = None
) -> None:
    """
    Run the TUI with an initialized VoiceAgent.

    NOTE: This is a text-mode interaction path; voice input loop is NOT started
    concurrently in this first iteration. Future versions will unify both.

    This function performs a late (re)import of textual so that environments
    where textual becomes available only after virtualenv / shell activation
    (e.g. Nix / devenv) still work even if the module was imported earlier
    before those paths were present.
    """
    # Late import / re-import path if we are still using stub objects
    if App is object:  # type: ignore
        try:
            # Dynamically import required modules
            from textual.app import App as RealApp, ComposeResult  # type: ignore
            from textual.containers import Horizontal, Vertical, Container  # type: ignore

            try:
                from textual.widgets import Static, Input, ScrollView  # type: ignore
            except ImportError:
                from textual.widgets import Static, Input  # type: ignore

                class ScrollView(Static):  # type: ignore
                    """Runtime fallback ScrollView if not provided by Textual."""

                    pass

            from textual.reactive import reactive  # type: ignore
            from textual.message import Message  # type: ignore
            from textual import events  # type: ignore

            # Inject real implementations into module globals so existing classes work
            globals().update(
                {
                    "App": RealApp,
                    "ComposeResult": ComposeResult,
                    "Horizontal": Horizontal,
                    "Vertical": Vertical,
                    "Container": Container,
                    "Static": Static,
                    "Input": Input,
                    "ScrollView": ScrollView,
                    "reactive": reactive,
                    "Message": Message,
                    "events": events,
                }
            )
        except Exception as imp_err:  # pragma: no cover
            raise RuntimeError(
                "Failed to import textual at runtime. Install dependencies first:\n"
                "  pip install textual rich\n"
                f"Underlying error: {imp_err}"
            ) from imp_err

    pipeline_status = pipeline_status or PipelineStatus()

    def _state_callback(component: str, state: str, message: Optional[str]) -> None:
        """Update PipelineStatus from core service callbacks."""
        state_map = {
            "idle": ComponentState.IDLE,
            "initializing": ComponentState.INITIALIZING,
            "ready": ComponentState.READY,
            "active": ComponentState.ACTIVE,
            "error": ComponentState.ERROR,
            "disabled": ComponentState.DISABLED,
        }
        comp_state = state_map.get(state.lower(), ComponentState.IDLE)
        field_map = {
            "audio_input": "audio_input",
            "audio_output": "audio_output",
            "stt": "stt",
            "llm": "llm",
            "tts": "tts",
        }
        target_field = field_map.get(component)
        if target_field and hasattr(pipeline_status, target_field):
            setattr(pipeline_status, target_field, comp_state)

        # Manage current message
        if comp_state in (ComponentState.ACTIVE, ComponentState.INITIALIZING):
            if message:
                pipeline_status.current_message = message
        elif comp_state == ComponentState.READY:
            # Clear transient message on READY
            pipeline_status.current_message = None

        # Manage error message
        if comp_state == ComponentState.ERROR:
            pipeline_status.error_message = message or "unknown error"
        else:
            if pipeline_status.error_message and comp_state in (
                ComponentState.ACTIVE,
                ComponentState.READY,
            ):
                # Clear previous error when component becomes active/ready again
                pipeline_status.error_message = None

    # Attach callback to voice agent if supported
    if hasattr(voice_agent, "set_state_callback"):
        try:
            voice_agent.set_state_callback(_state_callback)  # type: ignore[attr-defined]
        except Exception:
            # Non-fatal; TUI can still run
            pass

    ui_cfg = getattr(getattr(voice_agent, "config", None), "ui", None)
    adapter = AgentAdapter(voice_agent, pipeline_status=pipeline_status)
    app = VoiceAgentTUI(
        agent_adapter=adapter, pipeline_status=pipeline_status, ui_config=ui_cfg
    )

    # Prefer native async run if available to stay on main thread (avoids signal issues)
    if hasattr(app, "run_async"):
        await app.run_async()  # type: ignore
    else:  # Fallback: last resort executor (may have signal limitations)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, app.run)

    # Persist UI toggle changes back to config if possible
    try:
        cfg_path = getattr(voice_agent, "_config_path", None)
        if cfg_path:
            voice_agent.config.save(cfg_path)  # type: ignore[attr-defined]
    except Exception as _persist_err:  # pragma: no cover
        logging.getLogger(__name__).warning(
            f"Failed to persist UI config changes: {_persist_err}"
        )


# ---------------------------------------------------------------------------
# Optional convenience entry point
# ---------------------------------------------------------------------------


def launch_text_mode_agent(agent_factory: Callable[[], Any]) -> None:
    """
    Convenience synchronous launcher:
        from voice_agent.ui.tui_app import launch_text_mode_agent
        launch_text_mode_agent(lambda: VoiceAgent(config))
    """

    async def _inner():
        agent = agent_factory()
        await agent.initialize()
        await run_tui(agent)

    asyncio.run(_inner())


__all__ = [
    "ComponentState",
    "PipelineStatus",
    "ChatMessage",
    "ErrorBanner",
    "VoiceAgentTUI",
    "AgentAdapter",
    "run_tui",
    "launch_text_mode_agent",
]
