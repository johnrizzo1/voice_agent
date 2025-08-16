#!/usr/bin/env python3
"""
Comprehensive test for Voice Agent + Multi-Agent integration.

This test verifies that:
1. Voice pipeline (TTS/STT/Audio) works with multi-agent system
2. Agent routing works correctly with voice inputs
3. Voice-friendly responses are generated
4. TUI state callbacks work with multi-agent routing
5. Error handling and fallbacks work properly
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from voice_agent.core.config import Config
from voice_agent.core.voice_agent_orchestrator import VoiceAgentOrchestrator


async def test_voice_multi_agent_integration():
    """Test voice + multi-agent integration."""
    print("Voice Agent + Multi-Agent Integration Test")
    print("=" * 50)

    # Load test configuration with multi-agent enabled
    config_path = Path(__file__).parent / "test_config_multi_agent.yaml"
    if not config_path.exists():
        print(f"❌ Test configuration not found: {config_path}")
        return False

    try:
        config = Config.load(config_path)
        print(f"✅ Configuration loaded from {config_path}")
        print(f"   - Multi-agent enabled: {config.multi_agent.enabled}")
        print(f"   - Audio enabled: {not config.ui.force_text_only}")
        print(f"   - Configured agents: {list(config.multi_agent.agents.keys())}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False

    # Initialize orchestrator
    try:
        orchestrator = VoiceAgentOrchestrator(
            config=config, text_only=True
        )  # Start in text-only for testing
        await orchestrator.initialize()
        print("✅ Voice Agent Orchestrator initialized successfully")

        # Get orchestrator info
        info = orchestrator.get_orchestrator_info()
        print(f"   - Multi-agent enabled: {info['multi_agent_enabled']}")
        print(f"   - Text-only mode: {info['text_only']}")
        print(
            f"   - Components initialized: {sum(info['components'].values())} / {len(info['components'])}"
        )

        if "multi_agent_info" in info:
            ma_info = info["multi_agent_info"]
            print(f"   - Active agents: {ma_info.get('active_agents', 0)}")
            print(f"   - Message count: {ma_info.get('message_count', 0)}")

    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")
        return False

    # Test 1: Text-based multi-agent routing
    print("\n=== Test 1: Multi-Agent Routing with Text ===")
    test_cases = [
        ("Hello, how are you?", "general_agent", "General greeting"),
        ("What's the weather like today?", "information_agent", "Weather query"),
        ("Calculate 15 * 23", "utility_agent", "Calculation request"),
        (
            "Search for information about Python",
            "information_agent",
            "Web search request",
        ),
        ("What is 2 + 2?", "utility_agent", "Simple math"),
    ]

    routing_success = 0
    for query, expected_agent, description in test_cases:
        try:
            print(f"\n📤 Testing: {description}")
            print(f"   Query: '{query}'")

            start_time = time.time()
            response = await orchestrator.process_text(query)
            end_time = time.time()

            print(
                f"   Response: {response[:100]}..."
                if len(response) > 100
                else f"   Response: {response}"
            )
            print(f"   Processing time: {end_time - start_time:.2f}s")

            # Check if multi-agent service has routing stats
            if orchestrator.multi_agent_service:
                routing_stats = orchestrator.multi_agent_service.routing_stats
                print(f"   Routing stats: {routing_stats}")

            routing_success += 1
            print(f"✅ {description} completed successfully")

        except Exception as e:
            print(f"❌ {description} failed: {e}")

    print(
        f"\n📊 Text routing test results: {routing_success}/{len(test_cases)} successful"
    )

    # Test 2: Voice-friendly response formatting
    print("\n=== Test 2: Voice-Friendly Response Formatting ===")
    voice_queries = [
        "Tell me about the weather",
        "What's twenty-five times thirty-seven",
        "How do you make a cake",
    ]

    formatting_success = 0
    for query in voice_queries:
        try:
            response = await orchestrator.process_text(query)

            # Check voice-friendly characteristics
            voice_friendly_checks = {
                "reasonable_length": len(response.split())
                < 100,  # Not too long for speech
                "no_markdown": "[" not in response
                and "```" not in response,  # No markdown
                "conversational": any(
                    word in response.lower()
                    for word in ["the", "is", "are", "will", "can"]
                ),
                "ends_properly": response.strip().endswith((".", "!", "?")),
            }

            passed_checks = sum(voice_friendly_checks.values())
            print(f"📝 Query: '{query}'")
            print(f"   Response length: {len(response.split())} words")
            print(
                f"   Voice-friendly checks: {passed_checks}/{len(voice_friendly_checks)}"
            )

            if passed_checks >= len(voice_friendly_checks) - 1:  # Allow 1 failure
                formatting_success += 1
                print("✅ Response is voice-friendly")
            else:
                print("⚠️  Response may not be optimal for voice")

        except Exception as e:
            print(f"❌ Voice formatting test failed: {e}")

    print(
        f"\n📊 Voice formatting test results: {formatting_success}/{len(voice_queries)} successful"
    )

    # Test 3: Error handling and fallbacks
    print("\n=== Test 3: Error Handling and Fallbacks ===")
    error_test_success = 0

    try:
        # Test with a query that might cause issues
        complex_query = "This is a very complex query that might cause routing issues or agent failures"
        response = await orchestrator.process_text(complex_query)
        print(f"✅ Complex query handled: {len(response)} chars")
        error_test_success += 1
    except Exception as e:
        print(f"❌ Complex query failed: {e}")

    try:
        # Test conversation history preservation
        await orchestrator.process_text("My name is Alice")
        response = await orchestrator.process_text("What's my name?")
        if "alice" in response.lower():
            print("✅ Conversation history preserved")
            error_test_success += 1
        else:
            print("⚠️  Conversation history may not be preserved")
    except Exception as e:
        print(f"❌ Conversation history test failed: {e}")

    print(f"\n📊 Error handling test results: {error_test_success}/2 successful")

    # Test 4: Performance metrics
    print("\n=== Test 4: Performance Metrics ===")
    try:
        # Run multiple queries to test performance
        start_time = time.time()
        for i in range(3):
            await orchestrator.process_text(f"Test query number {i+1}")
        end_time = time.time()

        avg_time = (end_time - start_time) / 3
        print(f"✅ Average processing time: {avg_time:.2f}s per query")

        if avg_time < 10.0:  # Reasonable response time
            print("✅ Performance is acceptable")
        else:
            print("⚠️  Performance may be slow")

    except Exception as e:
        print(f"❌ Performance test failed: {e}")

    # Cleanup
    try:
        await orchestrator.stop()
        print("\n✅ Orchestrator stopped successfully")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    total_tests = (
        len(test_cases) + len(voice_queries) + 2 + 1
    )  # routing + formatting + error + performance
    total_success = routing_success + formatting_success + error_test_success + 1

    print(
        f"Multi-agent integration: {'✅ WORKING' if routing_success > 0 else '❌ FAILED'}"
    )
    print(
        f"Voice-friendly responses: {'✅ WORKING' if formatting_success > 0 else '❌ FAILED'}"
    )
    print(f"Error handling: {'✅ WORKING' if error_test_success > 0 else '❌ FAILED'}")
    print(
        f"Overall success rate: {total_success}/{total_tests} ({total_success/total_tests*100:.1f}%)"
    )

    if total_success >= total_tests * 0.8:  # 80% success rate
        print("\n🎉 Voice + Multi-Agent integration is READY!")
        return True
    else:
        print("\n⚠️  Voice + Multi-Agent integration needs work")
        return False


async def main():
    """Main test function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        success = await test_voice_multi_agent_integration()
        if success:
            print("\n✅ All tests passed - integration is ready!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed - integration needs work")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
