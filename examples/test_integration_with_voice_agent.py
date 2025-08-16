#!/usr/bin/env python3
"""
Test script to verify LlamaIndex integration with existing voice agent system.
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from voice_agent.core.config import Config, LLMConfig
from voice_agent.core.llamaindex_service import LlamaIndexService
from voice_agent.tools.registry import ToolRegistry


async def test_llamaindex_with_voice_agent_tools():
    """Test LlamaIndex with existing voice agent tools."""
    print("=" * 70)
    print("LlamaIndex + Voice Agent Tools Integration Test")
    print("=" * 70)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Use the same config as voice agent
    config = Config()
    llm_config = config.llm

    print(f"Using model: {llm_config.model}")
    print(f"Temperature: {llm_config.temperature}")
    print()

    # Initialize tool registry (same as voice agent)
    print("1. Initializing voice agent tool registry...")
    tool_registry = ToolRegistry()
    await tool_registry.initialize()

    available_tools = await tool_registry.get_available_tools()
    print(f"✅ Loaded {len(available_tools)} voice agent tools:")
    for tool in available_tools:
        print(f"   - {tool['name']}: {tool['description']}")
    print()

    # Initialize LlamaIndex service
    print("2. Initializing LlamaIndex service...")
    service = LlamaIndexService(llm_config)

    if not service.is_available:
        print("❌ LlamaIndex is not available!")
        return False

    await service.initialize()
    print("✅ LlamaIndex service initialized")
    print()

    # Create agent with voice agent tools
    print("3. Creating LlamaIndex agent with voice agent tools...")
    await service.create_agent(available_tools)

    info = await service.get_service_info()
    print(f"✅ Agent created with {info['num_tools']} tools")
    print()

    # Test with calculator (should work)
    print("4. Testing calculator tool...")
    calc_message = "Calculate 15 * 7 + 23"
    print(f"   User: {calc_message}")

    try:
        response = await service.chat(calc_message)
        print(f"   Agent: {response}")
        print()
    except Exception as e:
        print(f"   Error: {e}")
        print()

    # Test with weather tool (might work if API key available)
    print("5. Testing weather tool...")
    weather_message = "What's the weather like in London?"
    print(f"   User: {weather_message}")

    try:
        response = await service.chat(weather_message)
        print(f"   Agent: {response}")
        print()
    except Exception as e:
        print(f"   Error: {e}")
        print()

    # Test direct question (no tools needed)
    print("6. Testing direct question (no tools)...")
    direct_message = "What is the capital of France?"
    print(f"   User: {direct_message}")

    try:
        response = await service.chat(direct_message)
        print(f"   Agent: {response}")
        print()
    except Exception as e:
        print(f"   Error: {e}")
        print()

    # Cleanup
    print("7. Cleaning up...")
    await service.cleanup()
    print("✅ Cleanup completed")
    print()

    print("=" * 70)
    print("✅ LlamaIndex integration with Voice Agent tools tested!")
    print("=" * 70)
    return True


async def test_side_by_side_comparison():
    """Compare responses between original LLM service and LlamaIndex service."""
    print("=" * 70)
    print("Side-by-Side Comparison: Original LLM vs LlamaIndex")
    print("=" * 70)

    from voice_agent.core.llm_service import LLMService
    from voice_agent.tools.registry import ToolRegistry

    # Setup
    config = Config()
    llm_config = config.llm

    # Initialize tool registry
    tool_registry = ToolRegistry()
    await tool_registry.initialize()

    # Initialize original LLM service
    print("1. Initializing original LLM service...")
    original_llm = LLMService(llm_config)
    await original_llm.initialize()
    print("✅ Original LLM service ready")

    # Initialize LlamaIndex service
    print("2. Initializing LlamaIndex service...")
    llamaindex_llm = LlamaIndexService(llm_config)

    if not llamaindex_llm.is_available:
        print("❌ LlamaIndex not available, skipping comparison")
        return False

    await llamaindex_llm.initialize()
    await llamaindex_llm.create_agent()  # No tools for fair comparison
    print("✅ LlamaIndex service ready")
    print()

    # Test questions
    test_questions = [
        "What is 2 + 2?",
        "Explain what Python is in one sentence.",
        "What are the benefits of using local AI models?",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"{i}. Testing: '{question}'")
        print("-" * 50)

        # Original LLM
        print("Original LLM Service:")
        try:
            response1 = await original_llm.generate_response(question, [])
            print(f"   {response1}")
        except Exception as e:
            print(f"   Error: {e}")

        print()

        # LlamaIndex
        print("LlamaIndex Service:")
        try:
            response2 = await llamaindex_llm.chat(question)
            print(f"   {response2}")
        except Exception as e:
            print(f"   Error: {e}")

        print("\n" + "=" * 70 + "\n")

    # Cleanup
    await original_llm.cleanup()
    await llamaindex_llm.cleanup()

    print("Comparison completed!")
    return True


def main():
    """Main test function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("LlamaIndex Integration Test with Voice Agent")
        print("Usage: python test_integration_with_voice_agent.py [--comparison]")
        print()
        print("Options:")
        print("  --comparison    Run side-by-side comparison")
        print()
        print("This script tests:")
        print("1. Integration with voice agent tools")
        print("2. Tool execution via LlamaIndex agents")
        print("3. Compatibility with existing system")
        return

    async def run_tests():
        success = True

        # Main integration test
        success &= await test_llamaindex_with_voice_agent_tools()

        # Optional comparison test
        if len(sys.argv) > 1 and sys.argv[1] == "--comparison":
            print("\n")
            success &= await test_side_by_side_comparison()

        return success

    # Run the async tests
    try:
        success = asyncio.run(run_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
