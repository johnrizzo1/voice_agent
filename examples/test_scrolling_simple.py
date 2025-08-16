#!/usr/bin/env python3
"""
Simple Scrolling Test Application

This is a minimal test application to verify Textual ScrollView scrolling functionality.
It has only a content area and an input area. Every time you hit enter, it adds the
input content to the scrollable content area.

Usage:
    python examples/test_scrolling_simple.py

Controls:
    - Type text and press Enter to add to content
    - j/k keys: Manual scroll when content area is focused
    - Up/Down arrow keys: Manual scroll when content area is focused
    - Tab: Cycle focus between content and input areas
    - Ctrl+Q: Quit
"""

import asyncio
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, Static
from textual.binding import Binding
from textual.reactive import reactive
from textual import on
from textual.scroll_view import ScrollView
from textual.message import Message


class ScrollableContent(ScrollView):
    """A simple scrollable content area that can receive focus."""

    can_focus = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.content_lines = []
        self.line_count = 0

    def add_content(self, text: str):
        """Add a new line of content and auto-scroll to bottom."""
        self.line_count += 1
        new_line = f"{self.line_count:3d}: {text}"
        self.content_lines.append(new_line)

        # Update the content
        content_text = "\n".join(self.content_lines)

        # Clear and re-add content
        self.remove_children()
        content_widget = Static(content_text)
        self.mount(content_widget)

        # Force scroll to bottom
        self.scroll_end(animate=False)

        print(
            f"DEBUG: Added line {self.line_count}, total lines: {len(self.content_lines)}"
        )
        print(f"DEBUG: Scrolled to bottom, scroll_y: {self.scroll_y}")

    def on_key(self, event):
        """Handle key events when content area is focused."""
        if not self.has_focus:
            return

        if event.key == "j":
            # Scroll down (vi-style)
            self.scroll_relative(y=3, animate=True)
            event.prevent_default()
            print(f"DEBUG: Manual scroll down, new scroll_y: {self.scroll_y}")
        elif event.key == "k":
            # Scroll up (vi-style)
            self.scroll_relative(y=-3, animate=True)
            event.prevent_default()
            print(f"DEBUG: Manual scroll up, new scroll_y: {self.scroll_y}")
        elif event.key == "down":
            # Arrow key scroll down
            self.scroll_relative(y=1, animate=True)
            event.prevent_default()
            print(f"DEBUG: Arrow scroll down, new scroll_y: {self.scroll_y}")
        elif event.key == "up":
            # Arrow key scroll up
            self.scroll_relative(y=-1, animate=True)
            event.prevent_default()
            print(f"DEBUG: Arrow scroll up, new scroll_y: {self.scroll_y}")


class ScrollingTestApp(App):
    """Simple test app for scrolling functionality."""

    CSS = """
    ScrollableContent {
        border: solid $primary;
        height: 80%;
        padding: 1;
    }
    
    ScrollableContent:focus {
        border: solid $accent;
    }
    
    Input {
        height: 3;
        margin-top: 1;
        border: solid $secondary;
    }
    
    Input:focus {
        border: solid $accent;
    }
    
    .focused {
        border: solid $accent !important;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("tab", "cycle_focus", "Cycle Focus", priority=True),
        Binding("j", "scroll_down", "Scroll Down", priority=False),
        Binding("k", "scroll_up", "Scroll Up", priority=False),
        Binding("down", "scroll_down", "Scroll Down", priority=False),
        Binding("up", "scroll_up", "Scroll Up", priority=False),
    ]

    def __init__(self):
        super().__init__()
        self.content_area = None
        self.input_area = None

    def compose(self) -> ComposeResult:
        """Create the app layout."""
        with Vertical():
            self.content_area = ScrollableContent(id="content")
            self.input_area = Input(
                placeholder="Type something and press Enter...", id="input"
            )

            yield self.content_area
            yield self.input_area

        # Add initial content
        self.content_area.add_content("=== Scrolling Test Application ===")
        self.content_area.add_content("Type text below and press Enter to add it here.")
        self.content_area.add_content(
            "Focus the content area with Tab, then use j/k or arrow keys to scroll."
        )
        self.content_area.add_content(
            "Content will auto-scroll to bottom when new lines are added."
        )
        self.content_area.add_content("")

    async def on_mount(self) -> None:
        """Set initial focus to input area."""
        self.input_area.focus()

    @on(Input.Submitted)
    def handle_input_submit(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        text = event.value.strip()
        if text:
            self.content_area.add_content(text)
            self.input_area.value = ""  # Clear input

    def action_cycle_focus(self) -> None:
        """Cycle focus between content and input areas."""
        if self.content_area.has_focus:
            self.input_area.focus()
            print("DEBUG: Focus switched to input area")
        else:
            self.content_area.focus()
            print("DEBUG: Focus switched to content area")

    def action_scroll_down(self) -> None:
        """Scroll content area down (only if content area is focused)."""
        if self.content_area and self.content_area.has_focus:
            self.content_area.scroll_relative(y=3, animate=True)
            print(f"DEBUG: Action scroll down, scroll_y: {self.content_area.scroll_y}")

    def action_scroll_up(self) -> None:
        """Scroll content area up (only if content area is focused)."""
        if self.content_area and self.content_area.has_focus:
            self.content_area.scroll_relative(y=-3, animate=True)
            print(f"DEBUG: Action scroll up, scroll_y: {self.content_area.scroll_y}")


async def main():
    """Run the scrolling test application."""
    print("Starting Simple Scrolling Test Application...")
    print("=" * 60)
    print("Instructions:")
    print("1. Type text and press Enter to add it to the content area")
    print("2. Press Tab to switch focus between content and input areas")
    print("3. When content area is focused (blue border), use:")
    print("   - j/k keys to scroll down/up (vi-style)")
    print("   - Up/Down arrow keys to scroll")
    print("4. Content should auto-scroll to bottom when new lines are added")
    print("5. Press Ctrl+Q to quit")
    print("=" * 60)

    app = ScrollingTestApp()
    try:
        await app.run_async()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
