#!/usr/bin/env python3
"""
Test script for voice command quit functionality.
Tests the detect_voice_command and handle_voice_command methods for quit commands.
"""

import sys
import os

sys.path.insert(0, "src")

from voice_agent.ui.tui_app import VoiceAgentTUI


def test_quit_voice_command_detection():
    """Test that quit voice commands are properly detected."""
    print("Testing voice command detection for quit commands...")

    # Test various quit command phrases
    test_phrases = [
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
        # Test with natural speech variations
        "please end program",
        "I want to quit",
        "can you exit the application",
        "time to shutdown",
        # Test non-quit phrases to ensure they don't trigger
        "end of story",
        "quit complaining",
        "exit strategy",
    ]

    quit_commands = []
    non_quit_commands = []

    for phrase in test_phrases:
        result = VoiceAgentTUI.detect_voice_command(phrase)
        if result == "quit_app":
            quit_commands.append(phrase)
            print(f"✅ '{phrase}' -> detected as quit command")
        elif result is None:
            non_quit_commands.append(phrase)
            print(f"   '{phrase}' -> no command detected (expected for some phrases)")
        else:
            print(f"   '{phrase}' -> detected as '{result}' (not quit)")

    print(f"\nSummary:")
    print(f"  Phrases detected as quit commands: {len(quit_commands)}")
    print(f"  Phrases not detected as commands: {len(non_quit_commands)}")

    # Verify expected quit phrases are detected
    expected_quit_phrases = [
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

    detected_expected = [
        phrase for phrase in expected_quit_phrases if phrase in quit_commands
    ]
    print(
        f"  Expected quit phrases detected: {len(detected_expected)}/{len(expected_quit_phrases)}"
    )

    if len(detected_expected) == len(expected_quit_phrases):
        print("✅ All expected quit phrases were correctly detected!")
        return True
    else:
        missed = [
            phrase for phrase in expected_quit_phrases if phrase not in quit_commands
        ]
        print(f"❌ Missed quit phrases: {missed}")
        return False


def test_quit_command_patterns():
    """Test specific patterns that should trigger quit command."""
    print("\nTesting specific quit command patterns...")

    patterns_to_test = {
        # Core quit patterns from the code
        "end program": True,
        "quit": True,
        "exit": True,
        "shutdown": True,
        "close application": True,
        "terminate": True,
        "quit application": True,
        "exit application": True,
        "shut down": True,
        "close program": True,
        "terminate program": True,
        # Should NOT trigger quit
        "end the conversation": False,
        "quit talking": False,
        "exit the room": False,
        "program the system": False,
    }

    all_passed = True

    for phrase, should_quit in patterns_to_test.items():
        result = VoiceAgentTUI.detect_voice_command(phrase)
        is_quit = result == "quit_app"

        if is_quit == should_quit:
            status = "✅"
        else:
            status = "❌"
            all_passed = False

        expected = "should quit" if should_quit else "should NOT quit"
        actual = "detected as quit" if is_quit else "not detected as quit"
        print(f"{status} '{phrase}' - {expected}, {actual}")

    return all_passed


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING VOICE COMMAND QUIT FUNCTIONALITY")
    print("=" * 60)

    # Test 1: General quit command detection
    test1_passed = test_quit_voice_command_detection()

    # Test 2: Specific pattern testing
    test2_passed = test_quit_command_patterns()

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(
        f"Voice command detection test: {'✅ PASSED' if test1_passed else '❌ FAILED'}"
    )
    print(f"Quit pattern matching test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n🎉 ALL QUIT VOICE COMMAND TESTS PASSED!")
        print("The 'end program' voice command functionality is working correctly.")
    else:
        print("\n⚠️  SOME TESTS FAILED - Voice command detection needs attention.")

    exit(0 if (test1_passed and test2_passed) else 1)
