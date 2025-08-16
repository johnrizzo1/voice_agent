#!/usr/bin/env python3
"""
Test script to verify LlamaIndex + Ollama integration.
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from voice_agent.core.config import LLMConfig
from voice_agent.core.llamaindex_service import LlamaIndexService


async def test_llamaindex_ollama():
    """Test LlamaIndex with Ollama integration."""
    print("=" * 60)
    print("LlamaIndex + Ollama Integration Test")
    print("=" * 60)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create LLM config (using same defaults as voice agent)
    config = LLMConfig(
        provider="ollama", model="mistral:7b", temperature=0.7, max_tokens=512
    )

    print(f"Using model: {config.model}")
    print(f"Temperature: {config.temperature}")
    print()

    # Initialize LlamaIndex service
    print("1. Initializing LlamaIndex service...")
    service = LlamaIndexService(config)

    # Check if LlamaIndex is available
    if not service.is_available:
        print("❌ LlamaIndex is not available!")
        print("   Make sure LlamaIndex dependencies are installed:")
        print("   devenv shell")
        return False

    try:
        # Initialize the service
        await service.initialize()
        print("✅ LlamaIndex service initialized successfully")

        # Get service info
        info = await service.get_service_info()
        print(f"   Available: {info['available']}")
        print(f"   Initialized: {info['initialized']}")
        print(f"   Model: {info['model']}")
        print()

        # Test 2: Create a simple agent
        print("2. Creating ReAct agent...")
        await service.create_agent()

        updated_info = await service.get_service_info()
        if updated_info["has_agent"]:
            print("✅ ReAct agent created successfully")
        else:
            print("❌ Failed to create ReAct agent")
            return False
        print()

        # Test 3: Simple chat interaction
        print("3. Testing chat interaction...")
        test_message = "Hello! Can you tell me what 2+2 equals?"
        print(f"   User: {test_message}")

        response = await service.chat(test_message)
        print(f"   Agent: {response}")
        print()

        # Test 4: Test with a more complex query
        print("4. Testing complex reasoning...")
        complex_message = (
            "If I have 5 apples and give away 2, then buy 3 more, how many do I have?"
        )
        print(f"   User: {complex_message}")

        response = await service.chat(complex_message)
        print(f"   Agent: {response}")
        print()

        # Test 5: Vector index creation (empty)
        print("5. Testing vector index creation...")
        await service.create_vector_index()

        final_info = await service.get_service_info()
        if final_info["has_vector_index"]:
            print("✅ Vector index created successfully")
        else:
            print("❌ Failed to create vector index")
            return False
        print()

        # Cleanup
        print("6. Cleaning up...")
        await service.cleanup()
        print("✅ Cleanup completed")
        print()

        print("=" * 60)
        print("✅ ALL TESTS PASSED! LlamaIndex + Ollama integration is working.")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        return False


async def test_ollama_connection():
    """Test basic Ollama connection first."""
    print("Testing Ollama connection...")

    try:
        import ollama

        client = ollama.Client()

        # Test connection
        models = client.list()
        print(
            f"✅ Ollama is running. Available models: {len(models.get('models', []))}"
        )

        # Check if mistral:7b is available
        model_names = []
        if isinstance(models, dict) and "models" in models:
            model_names = [
                m.get("name", "unknown")
                for m in models["models"]
                if isinstance(m, dict)
            ]

        print(f"   Found models: {model_names}")

        if "mistral:7b" in model_names:
            print("✅ mistral:7b model is available")
            return True
        else:
            print("⚠️  mistral:7b model not found. Available models:")
            for name in model_names[:5]:  # Show first 5
                print(f"   - {name}")
            if len(model_names) > 5:
                print(f"   ... and {len(model_names)-5} more")

            # Try to pull the model
            print("Attempting to pull mistral:7b...")
            client.pull("mistral:7b")
            print("✅ mistral:7b model pulled successfully")
            return True

    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return False


def main():
    """Main test function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("LlamaIndex + Ollama Integration Test")
        print("Usage: python test_llamaindex_integration.py")
        print()
        print("This script tests:")
        print("1. LlamaIndex service initialization")
        print("2. Ollama LLM integration")
        print("3. ReAct agent creation")
        print("4. Basic chat functionality")
        print("5. Vector index creation")
        print()
        print("Prerequisites:")
        print("- Ollama running (ollama serve)")
        print("- mistral:7b model available")
        print("- LlamaIndex dependencies installed")
        return

    async def run_tests():
        # Test Ollama first
        if not await test_ollama_connection():
            return False
        print()

        # Then test LlamaIndex integration
        return await test_llamaindex_ollama()

    # Run the async tests
    try:
        success = asyncio.run(run_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
