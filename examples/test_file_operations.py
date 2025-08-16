#!/usr/bin/env python3
"""
Test script to verify file operations natural language mapping functionality.
"""

from src.voice_agent.tools.builtin.file_ops import FileOpsTool


def test_file_operations():
    """Test the file operations tool with natural language commands."""

    print("Testing File Operations Tool - Natural Language Mapping")
    print("=" * 60)

    # Initialize the tool
    file_tool = FileOpsTool()

    # Test content to save
    test_content = """This is a test file created by the voice agent.
Testing natural language file operations.
Created on: 2025-08-16
Status: SUCCESS"""

    # Test 1: Natural language "save" command
    print("\n1. Testing 'save this to test_file.txt'...")
    result1 = file_tool.execute("save", "test_file.txt", test_content)
    print(f"   Result: {result1}")

    # Test 2: Natural language "create" command
    print("\n2. Testing 'create file notes.md'...")
    result2 = file_tool.execute(
        "create file", "notes.md", "# Voice Agent Notes\n\nTesting file creation..."
    )
    print(f"   Result: {result2}")

    # Test 3: Natural language "open" command
    print("\n3. Testing 'open test_file.txt'...")
    result3 = file_tool.execute("open", "test_file.txt")
    print(f"   Result: {result3}")

    # Test 4: Natural language "view" command
    print("\n4. Testing 'view notes.md'...")
    result4 = file_tool.execute("view", "notes.md")
    print(f"   Result: {result4}")

    # Test 5: Check if files exist
    print("\n5. Testing 'check if test_file.txt exists'...")
    result5 = file_tool.execute("check", "test_file.txt")
    print(f"   Result: {result5}")

    # Test 6: List files
    print("\n6. Testing 'show files in current directory'...")
    result6 = file_tool.execute("show files", ".")
    print(f"   Result: {result6}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    successes = sum(
        1
        for r in [result1, result2, result3, result4, result5, result6]
        if r.get("success", False)
    )
    print(f"✅ Successful operations: {successes}/6")

    if successes == 6:
        print("🎉 All file operations working correctly!")
        return True
    else:
        print("❌ Some operations failed - check output above")
        return False


if __name__ == "__main__":
    success = test_file_operations()
    exit(0 if success else 1)
