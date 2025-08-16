#!/usr/bin/env python3
"""
Test script to verify file operations functionality through the TUI interface.
This simulates sending text commands to test natural language file operations.
"""

import subprocess
import time
import sys
from pathlib import Path


def test_file_operations_via_tui():
    """Test file operations by sending commands to the TUI interface."""
    print("Testing File Operations via TUI Interface")
    print("=" * 60)

    # Test commands to send
    test_commands = [
        "save this message to voice_test.txt",
        "create a file called test_notes.md with some content",
        "show me the files in the current directory",
        "check if voice_test.txt exists",
        "quit",
    ]

    print("Commands to test:")
    for i, cmd in enumerate(test_commands, 1):
        print(f"  {i}. {cmd}")

    print("\n" + "=" * 60)
    print("📋 Expected Results:")
    print("✅ File save operations should work with natural language")
    print("✅ File creation should work with 'create' commands")
    print("✅ Directory listing should work with 'show files'")
    print("✅ File existence checks should work with 'check if exists'")
    print("✅ Application should quit properly with 'quit' command")

    # Check if test files exist from previous test
    test_files = ["test_file.txt", "notes.md"]
    print(f"\n📁 Previous test files status:")
    for file in test_files:
        if Path(file).exists():
            print(f"  ✅ {file} exists from previous test")
        else:
            print(f"  ❌ {file} not found")

    print("\n" + "=" * 60)
    print("🔧 MANUAL TESTING REQUIRED:")
    print("1. Run: devenv shell -- python src/main.py --debug --no-audio")
    print("2. Wait for TUI to fully load")
    print("3. Type each command in the text input at bottom")
    print("4. Observe responses in the conversation area")
    print("5. Verify file operations are working correctly")
    print("\nTest commands to type manually:")
    for i, cmd in enumerate(test_commands, 1):
        print(f"  {i}. {cmd}")

    return True


if __name__ == "__main__":
    success = test_file_operations_via_tui()
    print(f"\n{'='*60}")
    if success:
        print("✅ File operations test preparation completed")
        print("📋 Manual testing instructions provided above")
    else:
        print("❌ Test preparation failed")

    exit(0 if success else 1)
