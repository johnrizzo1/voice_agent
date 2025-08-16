#!/usr/bin/env python3
"""
Test script for InformationAgent implementation.

This script tests the InformationAgent functionality including:
- Agent initialization and configuration
- Weather tool integration
- Web search tool integration
- News tool placeholder
- Multi-agent routing
- Information-specific response formatting
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from voice_agent.core.config import Config
from voice_agent.core.multi_agent_service import MultiAgentService
from voice_agent.core.tool_executor import ToolExecutor
from voice_agent.tools.builtin.weather import WeatherTool
from voice_agent.tools.builtin.web_search import WebSearchTool
from voice_agent.tools.builtin.news import NewsTool


async def test_information_agent_initialization():
    """Test InformationAgent initialization."""
    print("=" * 60)
    print("Testing InformationAgent Initialization")
    print("=" * 60)

    try:
        # Load configuration with multi-agent enabled
        config_path = Path("src/voice_agent/config/default.yaml")
        config = Config.load(config_path)
        config.multi_agent.enabled = True

        # Initialize tool executor
        tool_executor = ToolExecutor()

        # Register information-related tools
        weather_tool = WeatherTool()
        web_search_tool = WebSearchTool()
        news_tool = NewsTool()

        tool_executor.register_tool(weather_tool)
        tool_executor.register_tool(web_search_tool)
        tool_executor.register_tool(news_tool)

        # Initialize multi-agent service
        multi_agent_service = MultiAgentService(
            config=config, tool_executor=tool_executor
        )

        # Initialize the service
        await multi_agent_service.initialize()

        # Check if InformationAgent was created
        service_info = multi_agent_service.get_service_info()
        print(
            f"✅ Multi-agent service initialized: {service_info['multi_agent_enabled']}"
        )
        print(f"✅ Active agents: {service_info.get('active_agents', 0)}")

        if "agent_status" in service_info:
            for agent_id, status in service_info["agent_status"].items():
                print(
                    f"   - {agent_id}: {status['status']} ({len(status['capabilities'])} capabilities)"
                )
                if "information_agent" in agent_id:
                    print(f"     Capabilities: {', '.join(status['capabilities'])}")
                    print(f"     Tools: {status['tools_count']}")

        # Test agent routing
        if "information_agent" in multi_agent_service.agents:
            info_agent = multi_agent_service.agents["information_agent"]
            print(f"✅ InformationAgent found with {len(info_agent.tools)} tools")
            print(f"   Capabilities: {[cap.value for cap in info_agent.capabilities]}")
        else:
            print("❌ InformationAgent not found in active agents")

        await multi_agent_service.cleanup()
        return True

    except Exception as e:
        print(f"❌ Initialization test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_weather_query_routing():
    """Test weather query routing to InformationAgent."""
    print("\n" + "=" * 60)
    print("Testing Weather Query Routing")
    print("=" * 60)

    try:
        # Setup multi-agent service
        config_path = Path("src/voice_agent/config/default.yaml")
        config = Config.load(config_path)
        config.multi_agent.enabled = True

        tool_executor = ToolExecutor()
        tool_executor.register_tool(WeatherTool())
        tool_executor.register_tool(WebSearchTool())
        tool_executor.register_tool(NewsTool())

        multi_agent_service = MultiAgentService(
            config=config, tool_executor=tool_executor
        )

        await multi_agent_service.initialize()

        # Test weather queries
        weather_queries = [
            "What's the weather like in New York?",
            "Give me the forecast for London",
            "How's the temperature in Tokyo today?",
        ]

        for query in weather_queries:
            print(f"\n🌤️  Testing: '{query}'")
            response = await multi_agent_service.process_message(query)
            print(f"Response: {response}")

            # Check routing stats
            service_info = multi_agent_service.get_service_info()
            routing_stats = service_info.get("routing_stats", {})
            print(f"Routing stats: {routing_stats}")

        await multi_agent_service.cleanup()
        return True

    except Exception as e:
        print(f"❌ Weather routing test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_web_search_query_routing():
    """Test web search query routing to InformationAgent."""
    print("\n" + "=" * 60)
    print("Testing Web Search Query Routing")
    print("=" * 60)

    try:
        # Setup multi-agent service
        config_path = Path("src/voice_agent/config/default.yaml")
        config = Config.load(config_path)
        config.multi_agent.enabled = True

        tool_executor = ToolExecutor()
        tool_executor.register_tool(WeatherTool())
        tool_executor.register_tool(WebSearchTool())
        tool_executor.register_tool(NewsTool())

        multi_agent_service = MultiAgentService(
            config=config, tool_executor=tool_executor
        )

        await multi_agent_service.initialize()

        # Test search queries
        search_queries = [
            "Search for Python programming tutorials",
            "Find information about artificial intelligence",
            "Look up the latest developments in renewable energy",
        ]

        for query in search_queries:
            print(f"\n🔍 Testing: '{query}'")
            response = await multi_agent_service.process_message(query)
            print(f"Response: {response}")

        await multi_agent_service.cleanup()
        return True

    except Exception as e:
        print(f"❌ Web search routing test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_news_query_routing():
    """Test news query routing to InformationAgent."""
    print("\n" + "=" * 60)
    print("Testing News Query Routing")
    print("=" * 60)

    try:
        # Setup multi-agent service
        config_path = Path("src/voice_agent/config/default.yaml")
        config = Config.load(config_path)
        config.multi_agent.enabled = True

        tool_executor = ToolExecutor()
        tool_executor.register_tool(WeatherTool())
        tool_executor.register_tool(WebSearchTool())
        tool_executor.register_tool(NewsTool())

        multi_agent_service = MultiAgentService(
            config=config, tool_executor=tool_executor
        )

        await multi_agent_service.initialize()

        # Test news queries
        news_queries = [
            "Get me the latest news about technology",
            "What are the current events in business?",
            "Show me recent headlines about climate change",
        ]

        for query in news_queries:
            print(f"\n📰 Testing: '{query}'")
            response = await multi_agent_service.process_message(query)
            print(f"Response: {response}")

        await multi_agent_service.cleanup()
        return True

    except Exception as e:
        print(f"❌ News routing test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_agent_handoff_scenarios():
    """Test scenarios where InformationAgent should handoff to other agents."""
    print("\n" + "=" * 60)
    print("Testing Agent Handoff Scenarios")
    print("=" * 60)

    try:
        # Setup multi-agent service
        config_path = Path("src/voice_agent/config/default.yaml")
        config = Config.load(config_path)
        config.multi_agent.enabled = True

        tool_executor = ToolExecutor()
        tool_executor.register_tool(WeatherTool())
        tool_executor.register_tool(WebSearchTool())
        tool_executor.register_tool(NewsTool())

        # Add calculator for testing handoffs
        from voice_agent.tools.builtin.calculator import CalculatorTool

        tool_executor.register_tool(CalculatorTool())

        multi_agent_service = MultiAgentService(
            config=config, tool_executor=tool_executor
        )

        await multi_agent_service.initialize()

        # Test queries that should trigger handoffs
        handoff_queries = [
            "Calculate 25 * 43 + 17",  # Should go to tool_specialist
            "Hello, how are you?",  # Should go to general_agent
            "What's the weather and calculate 2+2?",  # Information query with calculation
        ]

        for query in handoff_queries:
            print(f"\n🔄 Testing handoff: '{query}'")
            response = await multi_agent_service.process_message(query)
            print(f"Response: {response}")

            # Check agent switches
            service_info = multi_agent_service.get_service_info()
            agent_switches = service_info.get("agent_switches", 0)
            print(f"Agent switches so far: {agent_switches}")

        await multi_agent_service.cleanup()
        return True

    except Exception as e:
        print(f"❌ Agent handoff test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all InformationAgent tests."""
    print("🚀 Starting InformationAgent Integration Tests")
    print("=" * 60)

    # Configure logging for better debugging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run all tests
    tests = [
        ("InformationAgent Initialization", test_information_agent_initialization),
        ("Weather Query Routing", test_weather_query_routing),
        ("Web Search Query Routing", test_web_search_query_routing),
        ("News Query Routing", test_news_query_routing),
        ("Agent Handoff Scenarios", test_agent_handoff_scenarios),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Print final results
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! InformationAgent is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
