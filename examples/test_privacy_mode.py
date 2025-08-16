#!/usr/bin/env python3
"""
Test script for privacy mode voice command functionality.
Tests the detect_voice_command and handle_voice_command methods for privacy commands.
"""

import sys

sys.path.insert(0, "src")

from voice_agent.ui.tui_app import VoiceAgentTUI


def test_privacy_voice_command_detection():
    """Test that privacy voice commands are properly detected."""
    print("Testing voice command detection for privacy mode commands...")

    # Test privacy mode commands
    privacy_on_phrases = ["privacy mode", "privacy mode on", "stop listening"]

    privacy_off_phrases = ["privacy mode off", "resume listening"]

    privacy_on_detected = []
    privacy_off_detected = []

    print("\nTesting Privacy Mode ON commands:")
    for phrase in privacy_on_phrases:
        result = VoiceAgentTUI.detect_voice_command(phrase)
        if result == "privacy_on":
            privacy_on_detected.append(phrase)
            print(f"✅ '{phrase}' -> detected as privacy_on command")
        else:
            print(f"❌ '{phrase}' -> detected as '{result}' (expected privacy_on)")

    print("\nTesting Privacy Mode OFF commands:")
    for phrase in privacy_off_phrases:
        result = VoiceAgentTUI.detect_voice_command(phrase)
        if result == "privacy_off":
            privacy_off_detected.append(phrase)
            print(f"✅ '{phrase}' -> detected as privacy_off command")
        else:
            print(f"❌ '{phrase}' -> detected as '{result}' (expected privacy_off)")

    print(f"\nSummary:")
    print(
        f"  Privacy ON phrases detected: {len(privacy_on_detected)}/{len(privacy_on_phrases)}"
    )
    print(
        f"  Privacy OFF phrases detected: {len(privacy_off_detected)}/{len(privacy_off_phrases)}"
    )

    all_passed = len(privacy_on_detected) == len(privacy_on_phrases) and len(
        privacy_off_detected
    ) == len(privacy_off_phrases)

    return all_passed


def test_dictation_voice_commands():
    """Test dictation mode voice commands."""
    print("\nTesting dictation mode voice commands...")

    dictation_commands = {
        "start dictation": "start_dictation",
        "take a dictation": "start_dictation",
        "begin dictation": "start_dictation",
        "end dictation": "end_dictation",
        "stop dictation": "end_dictation",
        "finish dictation": "end_dictation",
        "pause dictation": "pause_dictation",
        "resume dictation": "resume_dictation",
        "continue dictation": "resume_dictation",
        "cancel dictation": "cancel_dictation",
        "discard dictation": "cancel_dictation",
    }

    all_passed = True

    for phrase, expected_command in dictation_commands.items():
        result = VoiceAgentTUI.detect_voice_command(phrase)
        if result == expected_command:
            print(f"✅ '{phrase}' -> detected as {expected_command}")
        else:
            print(
                f"❌ '{phrase}' -> detected as '{result}' (expected {expected_command})"
            )
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING PRIVACY MODE & DICTATION VOICE COMMANDS")
    print("=" * 60)

    # Test 1: Privacy mode commands
    test1_passed = test_privacy_voice_command_detection()

    # Test 2: Dictation commands
    test2_passed = test_dictation_voice_commands()

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(
        f"Privacy mode voice commands: {'✅ PASSED' if test1_passed else '❌ FAILED'}"
    )
    print(f"Dictation voice commands: {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n🎉 ALL PRIVACY MODE & DICTATION TESTS PASSED!")
        print("Privacy mode toggle and dictation commands are working correctly.")
    else:
        print("\n⚠️  SOME TESTS FAILED - Voice command detection needs attention.")

    exit(0 if (test1_passed and test2_passed) else 1)
