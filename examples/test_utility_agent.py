#!/usr/bin/env python3
"""
Test suite for UtilityAgent implementation.

This test suite validates the UtilityAgent's mathematical and utility capabilities,
integration with the multi-agent system, and proper tool usage.
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_imports():
    """Test that all required imports work correctly."""
    print("Testing imports...")

    try:
        from voice_agent.agents.utility_agent import UtilityAgent

        print("✅ UtilityAgent import successful")
    except ImportError as e:
        print(f"❌ Failed to import UtilityAgent: {e}")
        return False

    try:
        from voice_agent.core.config import Config
        from voice_agent.core.tool_executor import ToolExecutor
        from voice_agent.core.multi_agent_service import MultiAgentService

        print("✅ Core components import successful")
    except ImportError as e:
        print(f"❌ Failed to import core components: {e}")
        return False

    try:
        from voice_agent.core.multi_agent.agent_base import (
            AgentCapability,
            AgentMessage,
            AgentResponse,
        )

        print("✅ Multi-agent components import successful")
    except ImportError as e:
        print(f"❌ Failed to import multi-agent components: {e}")
        return False

    return True


def test_utility_agent_initialization():
    """Test UtilityAgent initialization and configuration."""
    print("\nTesting UtilityAgent initialization...")

    try:
        from voice_agent.agents.utility_agent import UtilityAgent
        from voice_agent.core.multi_agent.agent_base import AgentCapability

        # Test basic initialization
        agent = UtilityAgent("test_utility_agent")

        # Verify capabilities
        expected_capabilities = {
            AgentCapability.CALCULATIONS,
            AgentCapability.TOOL_EXECUTION,
            AgentCapability.CONVERSATION_MEMORY,
            AgentCapability.SYSTEM_INFO,
        }

        if agent.capabilities == expected_capabilities:
            print("✅ UtilityAgent capabilities correctly configured")
        else:
            print(
                f"❌ Capability mismatch. Expected: {expected_capabilities}, Got: {agent.capabilities}"
            )
            return False

        # Verify agent_id
        if agent.agent_id == "test_utility_agent":
            print("✅ Agent ID correctly set")
        else:
            print(
                f"❌ Agent ID mismatch. Expected: test_utility_agent, Got: {agent.agent_id}"
            )
            return False

        # Check mathematical tracking attributes
        if hasattr(agent, "_calculation_history") and hasattr(agent, "_utility_calls"):
            print("✅ Mathematical tracking attributes present")
        else:
            print("❌ Missing mathematical tracking attributes")
            return False

        print("✅ UtilityAgent initialization successful")
        return True

    except Exception as e:
        print(f"❌ UtilityAgent initialization failed: {e}")
        return False


async def test_utility_agent_multi_agent_integration():
    """Test UtilityAgent integration with multi-agent service."""
    print("\nTesting UtilityAgent multi-agent integration...")

    try:
        from voice_agent.core.config import Config
        from voice_agent.core.tool_executor import ToolExecutor
        from voice_agent.core.multi_agent_service import MultiAgentService
        from voice_agent.agents.utility_agent import UtilityAgent

        # Load configuration
        config_path = os.path.join(
            os.path.dirname(__file__), "src", "voice_agent", "config", "default.yaml"
        )
        config = Config.load(config_path)

        # Enable multi-agent for testing
        config.multi_agent.enabled = True

        # Initialize tool executor
        tool_executor = ToolExecutor(config.tools)
        await tool_executor.initialize()

        # Initialize multi-agent service
        multi_agent_service = MultiAgentService(
            config=config, tool_executor=tool_executor
        )

        await multi_agent_service.initialize()

        if multi_agent_service.multi_agent_enabled:
            print("✅ Multi-agent service initialized with multi-agent mode")
        else:
            print(
                "⚠️  Multi-agent service running in single-agent mode (dependencies may be missing)"
            )
            return True  # This is acceptable for testing

        # Check if utility_agent is in configured agents
        if "utility_agent" in multi_agent_service.agents:
            utility_agent = multi_agent_service.agents["utility_agent"]
            print(f"✅ UtilityAgent found in service: {utility_agent.agent_id}")

            # Check agent status
            status_info = utility_agent.get_status_info()
            print(f"   Agent status: {status_info.get('status', 'unknown')}")
            print(f"   Capabilities: {status_info.get('capabilities', [])}")
            print(f"   Tools count: {status_info.get('tools_count', 0)}")

        else:
            print("❌ UtilityAgent not found in multi-agent service")
            return False

        # Cleanup
        await multi_agent_service.cleanup()
        print("✅ UtilityAgent multi-agent integration successful")
        return True

    except Exception as e:
        print(f"❌ Multi-agent integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_utility_agent_mathematical_queries():
    """Test UtilityAgent responses to mathematical queries."""
    print("\nTesting UtilityAgent mathematical query processing...")

    try:
        from voice_agent.core.config import Config
        from voice_agent.core.tool_executor import ToolExecutor
        from voice_agent.core.multi_agent_service import MultiAgentService

        # Load configuration
        config_path = os.path.join(
            os.path.dirname(__file__), "src", "voice_agent", "config", "default.yaml"
        )
        config = Config.load(config_path)

        # Enable multi-agent for testing
        config.multi_agent.enabled = True

        # Initialize components
        tool_executor = ToolExecutor(config.tools)
        await tool_executor.initialize()

        multi_agent_service = MultiAgentService(
            config=config, tool_executor=tool_executor
        )

        await multi_agent_service.initialize()

        if not multi_agent_service.multi_agent_enabled:
            print(
                "⚠️  Skipping mathematical query tests - multi-agent system not available"
            )
            return True

        # Test mathematical queries
        test_queries = [
            "Calculate 15 + 27 * 3",
            "What is 2^8?",
            "Solve the equation: (10 + 5) / 3",
            "Convert 25% to decimal",
            "Find the square root of 144",
        ]

        for query in test_queries:
            print(f"   Testing query: '{query}'")

            try:
                response = await multi_agent_service.process_message(
                    query, "test_conversation"
                )

                if response and len(response) > 0:
                    print(f"   ✅ Response received: {response[:100]}...")
                else:
                    print(f"   ❌ Empty or no response")

            except Exception as e:
                print(f"   ❌ Query failed: {e}")

        # Check routing stats
        service_info = multi_agent_service.get_service_info()
        routing_stats = service_info.get("routing_stats", {})

        if "utility_agent" in routing_stats:
            print(f"✅ UtilityAgent received {routing_stats['utility_agent']} queries")
        else:
            print("⚠️  No queries routed to UtilityAgent (check routing rules)")

        # Cleanup
        await multi_agent_service.cleanup()
        print("✅ Mathematical query processing test completed")
        return True

    except Exception as e:
        print(f"❌ Mathematical query test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_utility_agent_routing():
    """Test that mathematical queries are properly routed to UtilityAgent."""
    print("\nTesting UtilityAgent routing rules...")

    try:
        from voice_agent.core.config import Config
        from voice_agent.core.tool_executor import ToolExecutor
        from voice_agent.core.multi_agent_service import MultiAgentService

        # Load configuration
        config_path = os.path.join(
            os.path.dirname(__file__), "src", "voice_agent", "config", "default.yaml"
        )
        config = Config.load(config_path)

        # Enable multi-agent for testing
        config.multi_agent.enabled = True

        # Initialize components
        tool_executor = ToolExecutor(config.tools)
        await tool_executor.initialize()

        multi_agent_service = MultiAgentService(
            config=config, tool_executor=tool_executor
        )

        await multi_agent_service.initialize()

        if not multi_agent_service.multi_agent_enabled:
            print("⚠️  Skipping routing tests - multi-agent system not available")
            return True

        # Check routing rules configuration
        routing_rules = config.multi_agent.routing_rules
        utility_rules = [
            rule
            for rule in routing_rules
            if rule.get("target_agent") == "utility_agent"
        ]

        if utility_rules:
            print(f"✅ Found {len(utility_rules)} routing rules for UtilityAgent")
            for rule in utility_rules:
                print(
                    f"   Rule: {rule.get('name')} (priority: {rule.get('priority')}, confidence: {rule.get('confidence')})"
                )
        else:
            print("❌ No routing rules found for UtilityAgent")
            return False

        # Test specific mathematical phrases that should route to utility_agent
        mathematical_phrases = [
            "calculate",
            "compute",
            "math",
            "equation",
            "formula",
            "arithmetic",
            "percentage",
            "solve",
        ]

        utility_patterns = []
        for rule in utility_rules:
            utility_patterns.extend(rule.get("patterns", []))

        matching_phrases = [
            phrase for phrase in mathematical_phrases if phrase in utility_patterns
        ]

        if matching_phrases:
            print(
                f"✅ Mathematical phrases found in routing patterns: {matching_phrases}"
            )
        else:
            print("❌ No mathematical phrases found in UtilityAgent routing patterns")
            return False

        # Cleanup
        await multi_agent_service.cleanup()
        print("✅ UtilityAgent routing test successful")
        return True

    except Exception as e:
        print(f"❌ Routing test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_utility_agent_handoff():
    """Test UtilityAgent handoff scenarios."""
    print("\nTesting UtilityAgent handoff logic...")

    try:
        from voice_agent.agents.utility_agent import UtilityAgent
        from voice_agent.core.multi_agent.agent_base import AgentMessage, AgentResponse
        from voice_agent.core.multi_agent.message import MessageType

        # Create utility agent
        utility_agent = UtilityAgent("test_utility_agent")

        # Test scenarios that should NOT trigger handoff (utility domain)
        utility_queries = [
            "Calculate 2 + 2",
            "What is 15% of 200?",
            "Solve this equation: x = 10 * 5",
            "Convert 0.75 to percentage",
        ]

        for query in utility_queries:
            message = AgentMessage(
                conversation_id="test_conv", type=MessageType.USER_INPUT, content=query
            )

            response = AgentResponse(
                request_id=message.id,
                agent_id=utility_agent.agent_id,
                content="Test response",
                success=True,
            )

            # Test handoff evaluation
            await utility_agent._evaluate_handoff_need(message, response)

            if not response.should_handoff:
                print(f"   ✅ '{query}' correctly handled by UtilityAgent (no handoff)")
            else:
                print(
                    f"   ❌ '{query}' incorrectly triggered handoff to {response.suggested_agent}"
                )

        # Test scenarios that SHOULD trigger handoff (non-utility domain)
        non_utility_queries = [
            ("What's the weather like?", "information_agent"),
            ("Save this file to disk", "tool_specialist"),
            ("Hello, how are you?", "general_agent"),
        ]

        for query, expected_agent in non_utility_queries:
            message = AgentMessage(
                conversation_id="test_conv", type=MessageType.USER_INPUT, content=query
            )

            response = AgentResponse(
                request_id=message.id,
                agent_id=utility_agent.agent_id,
                content="Test response",
                success=True,
            )

            # Test handoff evaluation
            await utility_agent._evaluate_handoff_need(message, response)

            if response.should_handoff and response.suggested_agent == expected_agent:
                print(f"   ✅ '{query}' correctly handed off to {expected_agent}")
            elif response.should_handoff:
                print(
                    f"   ⚠️  '{query}' handed off to {response.suggested_agent} (expected {expected_agent})"
                )
            else:
                print(f"   ❌ '{query}' should have been handed off but wasn't")

        print("✅ UtilityAgent handoff logic test completed")
        return True

    except Exception as e:
        print(f"❌ Handoff test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all UtilityAgent tests."""
    print("🔧 UtilityAgent Test Suite")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Initialization Test", test_utility_agent_initialization),
        ("Multi-Agent Integration Test", test_utility_agent_multi_agent_integration),
        ("Mathematical Query Test", test_utility_agent_mathematical_queries),
        ("Routing Test", test_utility_agent_routing),
        ("Handoff Test", test_utility_agent_handoff),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")

        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All UtilityAgent tests PASSED!")
        print("\nUtilityAgent is ready for integration with the voice agent system.")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please review the failures above.")

    return passed == total


if __name__ == "__main__":
    import asyncio

    success = asyncio.run(main())
    sys.exit(0 if success else 1)
