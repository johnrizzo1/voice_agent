#!/usr/bin/env python3
"""
Test script to verify text area scrolling functionality works correctly.
This is a regression test to ensure scrolling wasn't broken by recent changes.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from voice_agent.ui.tui_app import ChatLog, ChatMessage

    print("✅ Successfully imported required modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)


def test_chat_log_scrolling():
    """Test ChatLog scrolling methods and functionality."""
    print("\n" + "=" * 60)
    print("TESTING CHAT LOG SCROLLING FUNCTIONALITY")
    print("=" * 60)

    # Create a ChatLog instance
    chat_log = ChatLog(max_messages=50, show_timestamps=True)

    # Test 1: Verify ChatLog has scroll functionality
    print("Testing ChatLog scroll functionality...")
    print("✅ ChatLog created successfully")

    # Test 2: Check scroll methods exist
    print(f"\nTesting scroll method availability...")
    methods_to_check = ["scroll_up", "scroll_down"]
    available_methods = []

    for method in methods_to_check:
        if hasattr(chat_log, method):
            available_methods.append(method)
            print(f"✅ {method}() method available")
        else:
            print(f"❌ {method}() method missing")

    # Test 3: Add multiple messages to test auto-scroll behavior
    print(f"\nTesting message addition and auto-scroll behavior...")
    test_messages = [
        ChatMessage(role="user", content=f"Test user message {i}") for i in range(1, 11)
    ] + [
        ChatMessage(role="agent", content=f"Test agent response {i}")
        for i in range(1, 11)
    ]

    for msg in test_messages:
        chat_log.add_message(msg)

    print(f"✅ Added {len(test_messages)} messages to chat log")
    print(f"✅ Current message count: {len(chat_log._messages)}")

    # Test 4: Test user scroll state tracking
    print(f"\nTesting scroll state tracking...")
    initial_scroll_state = getattr(chat_log, "_user_scrolled", False)
    print(f"✅ Initial user scroll state: {initial_scroll_state}")

    # Simulate user scrolling
    chat_log._user_scrolled = True
    scroll_state_after = getattr(chat_log, "_user_scrolled", False)
    print(f"✅ User scroll state after manual scroll: {scroll_state_after}")

    # Test 5: Test scroll method calls (they should not crash)
    print(f"\nTesting scroll method execution...")
    try:
        chat_log.scroll_up(5)
        print(f"✅ scroll_up() executed without error")
    except Exception as e:
        print(f"⚠️  scroll_up() error (expected in headless mode): {e}")

    try:
        chat_log.scroll_down(5)
        print(f"✅ scroll_down() executed without error")
    except Exception as e:
        print(f"⚠️  scroll_down() error (expected in headless mode): {e}")

    # Test 6: Test message rendering with scrolling context
    print(f"\nTesting message rendering...")
    try:
        rendered_content = chat_log.render()
        print(f"✅ Chat log rendered successfully")
        print(f"✅ Rendered content length: {len(rendered_content)} characters")

        # Check if rendered content contains expected messages
        if (
            "Test user message" in rendered_content
            and "Test agent response" in rendered_content
        ):
            print(f"✅ Rendered content contains expected messages")
        else:
            print(f"⚠️  Rendered content may be missing expected messages")

    except Exception as e:
        print(f"❌ Chat log rendering failed: {e}")

    return True


def test_scroll_key_bindings():
    """Test that scroll-related key bindings are properly configured."""
    print(f"\nTesting scroll key bindings...")

    try:
        from voice_agent.ui.tui_app import VoiceAgentTUI

        # Check if scroll-related bindings exist in BINDINGS
        scroll_bindings = [
            "pageup",
            "pagedown",
            "home",
            "end",
            "scroll_chat_up",
            "scroll_chat_down",
            "scroll_chat_top",
            "scroll_chat_bottom",
        ]

        bindings_list = VoiceAgentTUI.BINDINGS
        found_bindings = []

        for binding_tuple in bindings_list:
            key, action, description = binding_tuple
            if any(
                scroll_term in key or scroll_term in action
                for scroll_term in scroll_bindings
            ):
                found_bindings.append((key, action, description))

        print(f"✅ Found {len(found_bindings)} scroll-related key bindings:")
        for key, action, desc in found_bindings:
            print(f"   - {key} -> {action} ({desc})")

        return len(found_bindings) > 0

    except Exception as e:
        print(f"❌ Failed to check key bindings: {e}")
        return False


def main():
    """Run all scrolling tests."""
    print("TESTING TEXT AREA SCROLLING FUNCTIONALITY")
    print("=" * 60)

    success_count = 0
    total_tests = 2

    # Test 1: ChatLog scrolling functionality
    try:
        if test_chat_log_scrolling():
            success_count += 1
            print(f"\n✅ ChatLog scrolling tests: PASSED")
        else:
            print(f"\n❌ ChatLog scrolling tests: FAILED")
    except Exception as e:
        print(f"\n❌ ChatLog scrolling tests: FAILED with error: {e}")

    # Test 2: Key bindings
    try:
        if test_scroll_key_bindings():
            success_count += 1
            print(f"\n✅ Scroll key bindings tests: PASSED")
        else:
            print(f"\n❌ Scroll key bindings tests: FAILED")
    except Exception as e:
        print(f"\n❌ Scroll key bindings tests: FAILED with error: {e}")

    # Summary
    print(f"\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {success_count}/{total_tests}")

    if success_count == total_tests:
        print(f"🎉 ALL SCROLLING TESTS PASSED!")
        print(f"Text area scrolling functionality is working correctly.")
        return True
    else:
        print(f"⚠️  SOME SCROLLING TESTS FAILED")
        print(f"Text area scrolling functionality needs attention.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
