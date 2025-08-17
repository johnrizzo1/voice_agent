#!/usr/bin/env python3
"""
Simple scrolling test application.

This app creates a basic content area and input area to test auto-scrolling
behavior when new content is added. It's designed to isolate scrolling issues
from the more complex voice agent TUI.

Usage:
    python examples/test_scrolling_simple.py

Controls:
    - Type in the input box and press Enter to add content
    - Content should automatically scroll to show new additions
    - Esc to clear input, Ctrl+Q to quit
"""

import asyncio
from datetime import datetime
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Input

# Try to import ScrollView from the correct location
try:
    from textual.widgets import ScrollView
except ImportError:
    try:
        from textual.containers import ScrollView  # Some versions have it here
    except ImportError:
        # Fallback - create a ScrollView-like class based on Static
        class ScrollView(Static):
            def scroll_end(self, animate=False):
                """Fallback scroll_end implementation."""
                try:
                    if hasattr(self, "scroll_y") and hasattr(self, "max_scroll_y"):
                        self.scroll_y = getattr(self, "max_scroll_y", 0)
                    elif hasattr(self, "scroll_to"):
                        self.scroll_to(y=999999)
                    elif hasattr(self, "scroll_relative"):
                        self.scroll_relative(y=1000)
                except Exception:
                    pass


from textual import events


class ContentArea(ScrollView):
    """Properly implemented ScrollView for displaying messages with auto-scroll."""

    def __init__(self):
        super().__init__()
        self.messages = []
        self.message_count = 0
        self.content_widget = Static("")

    def compose(self) -> ComposeResult:
        """Compose the ScrollView with a Static widget inside."""
        yield self.content_widget

    def add_content(self, content: str) -> None:
        """Add new content and auto-scroll to bottom."""
        self.message_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.messages.append(f"[{timestamp}] Message {self.message_count}: {content}")

        # Update the display and auto-scroll
        self._update_display()

    def _update_display(self) -> None:
        """Update the display content and auto-scroll to bottom."""
        if not self.messages:
            content = "Content area - type something below and press Enter to add content.\n\nThis area should automatically scroll to show new content when it's added."
        else:
            # Show all messages
            content = "\n".join(self.messages)

            # Add some visual padding
            content += "\n\n" + "─" * 50 + "\n"
            content += f"Total messages: {self.message_count}\n"
            content += "Latest content should be visible above"

        # Update the content widget
        self.content_widget.update(content)

        # Auto-scroll to bottom - use the proper ScrollView method
        self.scroll_end(animate=False)


class InputArea(Horizontal):
    """Simple input area for entering content."""

    def __init__(self):
        super().__init__()
        self.input_field = Input(
            placeholder="Type your message here and press Enter..."
        )

    def compose(self) -> ComposeResult:
        yield Static("Input: ", classes="input-label")
        yield self.input_field

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        content = event.value.strip()
        if content:
            # Send message to parent app
            self.post_message(ContentSubmitted(content))
            # Clear the input
            self.input_field.value = ""


class ContentSubmitted(events.Message):
    """Message sent when content is submitted."""

    def __init__(self, content: str):
        self.content = content
        super().__init__()


class ScrollTestApp(App):
    """Simple test app for scrolling behavior."""

    CSS = """
    Screen {
        layout: vertical;
    }
    
    #content-area {
        height: 1fr;
        border: solid $primary;
        padding: 1;
        margin: 1;
    }
    
    #input-area {
        height: 3;
        border: solid $accent;
        padding: 0 1;
        margin: 0 1 1 1;
    }
    
    .input-label {
        width: 8;
        padding: 0 1;
        content-align: center middle;
    }
    
    #instructions {
        height: 3;
        background: $surface-lighten-1;
        padding: 1;
        text-align: center;
    }
    
    #status {
        height: 1;
        background: $surface-darken-1;
        padding: 0 1;
        text-align: center;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("escape", "clear_input", "Clear Input"),
        ("up", "scroll_up", "Scroll Up"),
        ("down", "scroll_down", "Scroll Down"),
        ("ctrl+t", "add_test_content", "Add Test Content"),
    ]

    def __init__(self):
        super().__init__()
        self.content_area = ContentArea()
        self.input_area = InputArea()

    def compose(self) -> ComposeResult:
        """Create the app layout."""
        instructions = Static(
            "SCROLL TEST APP\n"
            "Type messages below and press Enter. The content area should auto-scroll to show new messages.\n"
            "Press Ctrl+Q to quit, Esc to clear input, Ctrl+T to add test content quickly."
        )
        instructions.id = "instructions"

        self.content_area.id = "content-area"
        self.input_area.id = "input-area"

        status = Static("Status: Ready - Try typing a message and pressing Enter")
        status.id = "status"

        yield instructions
        yield self.content_area
        yield self.input_area
        yield status

    async def on_mount(self) -> None:
        """Focus the input field when the app starts."""
        try:
            # Try to focus the input field - different Textual versions handle this differently
            focus_result = self.set_focus(self.input_area.input_field)
            if focus_result is not None:
                await focus_result
        except Exception:
            # Fallback - try without await
            try:
                self.set_focus(self.input_area.input_field)
            except Exception:
                pass

        # Add some initial content to demonstrate scrolling
        initial_messages = [
            "Welcome to the scroll test!",
            "This content area should automatically scroll to the bottom when new content is added.",
            "Try typing some messages below to test the scrolling behavior.",
            "The content area should always show the latest message after adding new content.",
            "You can also press Ctrl+T to add test content quickly.",
            "If scrolling works correctly, you should see this message and the status below it.",
            "This is message 7 to test scrolling beyond visible area.",
            "This is message 8 - if you can see this, scrolling is working!",
            "Message 9 - the scroll should automatically show the latest content.",
            "Final initial message 10 - this should be visible at the bottom.",
        ]

        for i, msg in enumerate(initial_messages):
            self.content_area.add_content(msg)
            # Small delay to make the scrolling more visible
            if i < len(initial_messages) - 1:  # Don't delay on the last message
                await asyncio.sleep(0.1)

        # Update status
        self.query_one("#status").update(
            "Status: Initial content loaded - ready for input"
        )

    async def on_content_submitted(self, event: ContentSubmitted) -> None:
        """Handle new content submission."""
        self.content_area.add_content(event.content)

        # Update status
        self.query_one("#status").update(
            f"Status: Added message {self.content_area.message_count}"
        )

        # If we have many messages, add a separator occasionally
        if self.content_area.message_count % 5 == 0:
            await asyncio.sleep(0.1)  # Brief pause
            self.content_area.add_content(
                f"--- Milestone: {self.content_area.message_count} messages added ---"
            )

    async def action_clear_input(self) -> None:
        """Clear the input field."""
        self.input_area.input_field.value = ""
        try:
            focus_result = self.set_focus(self.input_area.input_field)
            if focus_result is not None:
                await focus_result
        except Exception:
            try:
                self.set_focus(self.input_area.input_field)
            except Exception:
                pass
        self.query_one("#status").update("Status: Input cleared")

    async def action_add_test_content(self) -> None:
        """Add test content quickly."""
        test_contents = [
            "This is test content line 1",
            "This is test content line 2",
            "This is test content line 3",
            "Auto-scroll should work for these too!",
            "Adding more content to test scrolling...",
            "Does the content area scroll to show this?",
            "Final test message - you should see this at the bottom!",
        ]

        for content in test_contents:
            self.content_area.add_content(content)
            await asyncio.sleep(0.1)

        self.query_one("#status").update("Status: Test content added")

    async def action_scroll_up(self) -> None:
        """Manually scroll up in content area."""
        self.content_area.scroll_relative(y=-3)
        self.query_one("#status").update("Status: Scrolled up manually")

    async def action_scroll_down(self) -> None:
        """Manually scroll down in content area."""
        self.content_area.scroll_relative(y=3)
        self.query_one("#status").update("Status: Scrolled down manually")

    async def on_key(self, event: events.Key) -> None:
        """Handle global key events."""
        if event.key == "ctrl+q":
            self.exit()


def main():
    """Run the scroll test app."""
    print("Starting scroll test app...")
    print("If you see import errors, make sure you're running in the devenv shell:")
    print("  devenv shell -- python examples/test_scrolling_simple.py")
    print()

    app = ScrollTestApp()
    app.run()


if __name__ == "__main__":
    main()
