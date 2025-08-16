#!/usr/bin/env python3
"""
Test script for ProductivityAgent integration with the multi-agent voice system.

This script tests:
1. ProductivityAgent initialization and configuration
2. File operations routing and processing
3. Calendar operations routing and processing
4. Multi-agent service integration
5. Context preservation and error handling
6. Tool integration (file_ops, calendar)
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    from voice_agent.core.config import Config
    from voice_agent.core.tool_executor import ToolExecutor
    from voice_agent.core.multi_agent_service import MultiAgentService
    from voice_agent.tools.registry import ToolRegistry

    # Test imports for multi-agent components
    from voice_agent.agents import ProductivityAgent
    from voice_agent.core.multi_agent.agent_base import AgentCapability
    from voice_agent.tools.builtin.file_ops import FileOpsTool
    from voice_agent.tools.builtin.calendar import CalendarTool

    print("✅ All imports successful")
    IMPORTS_AVAILABLE = True

except ImportError as e:
    print(f"❌ Import error: {e}")
    IMPORTS_AVAILABLE = False


async def test_productivity_agent_initialization():
    """Test ProductivityAgent initialization and basic functionality."""
    print("\n🧪 Testing ProductivityAgent Initialization...")

    try:
        # Create a basic agent config
        agent_config = type(
            "Config",
            (),
            {
                "agent_id": "test_productivity_agent",
                "capabilities": [
                    AgentCapability.FILE_OPERATIONS,
                    AgentCapability.CALENDAR_MANAGEMENT,
                    AgentCapability.TOOL_EXECUTION,
                    AgentCapability.TASK_PLANNING,
                ],
                "system_prompt": "Test productivity agent",
                "max_concurrent_tasks": 3,
                "timeout_seconds": 30.0,
                "metadata": {"context_window": 2048},
            },
        )()

        # Initialize ProductivityAgent
        agent = ProductivityAgent(
            agent_id="test_productivity_agent",
            config=agent_config,
            llm_config={"model": "mistral:7b", "temperature": 0.7},
        )

        print(f"✅ ProductivityAgent initialized: {agent.agent_id}")
        print(f"   Capabilities: {[cap.value for cap in agent.capabilities]}")
        print(f"   Status: {agent.status.value}")

        # Test capability checking
        assert agent.can_handle_capability(AgentCapability.FILE_OPERATIONS)
        assert agent.can_handle_capability(AgentCapability.CALENDAR_MANAGEMENT)
        assert agent.can_handle_capability(AgentCapability.TASK_PLANNING)
        print("✅ Capability checking works correctly")

        return True

    except Exception as e:
        print(f"❌ ProductivityAgent initialization failed: {e}")
        return False


async def test_tool_integration():
    """Test integration of file_ops and calendar tools."""
    print("\n🧪 Testing Tool Integration...")

    try:
        # Test FileOpsTool
        file_tool = FileOpsTool()
        help_info = file_tool.get_help()
        print(f"✅ FileOpsTool initialized: {help_info['name']}")
        print(f"   Operations: {list(help_info['operations'].keys())}")

        # Test CalendarTool
        calendar_tool = CalendarTool()
        calendar_help = calendar_tool.get_help()
        print(f"✅ CalendarTool initialized: {calendar_help['name']}")
        print(f"   Operations: {list(calendar_help['operations'].keys())}")

        # Test basic tool execution
        # File operations test
        result = file_tool.execute("exists", "/tmp")
        print(f"✅ File tool test result: {result['success']}")

        # Calendar operations test
        calendar_result = calendar_tool.execute("list_events")
        print(f"✅ Calendar tool test result: {calendar_result['success']}")

        return True

    except Exception as e:
        print(f"❌ Tool integration test failed: {e}")
        return False


async def test_multi_agent_service_integration():
    """Test ProductivityAgent integration with MultiAgentService."""
    print("\n🧪 Testing Multi-Agent Service Integration...")

    try:
        # Load configuration
        config_path = project_root / "src" / "voice_agent" / "config" / "default.yaml"
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return False

        config = Config.load(config_path)

        # Enable multi-agent mode for testing
        config.multi_agent.enabled = True
        print("✅ Configuration loaded and multi-agent enabled")

        # Initialize tool executor with config
        from voice_agent.core.config import ToolsConfig

        tools_config = ToolsConfig(
            enabled=[
                "calculator",
                "weather",
                "file_ops",
                "calendar",
                "web_search",
                "news",
            ]
        )
        tool_executor = ToolExecutor(tools_config)
        print("✅ Tool executor initialized")

        # Initialize multi-agent service
        multi_agent_service = MultiAgentService(
            config=config, tool_executor=tool_executor
        )

        await multi_agent_service.initialize()

        service_info = multi_agent_service.get_service_info()
        print(
            f"✅ Multi-agent service initialized: {service_info['multi_agent_enabled']}"
        )
        print(f"   Active agents: {service_info.get('active_agents', 0)}")
        print(
            f"   Available agents: {list(service_info.get('agent_status', {}).keys())}"
        )

        # Check if ProductivityAgent is loaded
        if "productivity_agent" in service_info.get("agent_status", {}):
            print("✅ ProductivityAgent found in active agents")
            agent_status = service_info["agent_status"]["productivity_agent"]
            print(f"   Agent status: {agent_status['status']}")
            print(f"   Capabilities: {agent_status['capabilities']}")
            print(f"   Tools count: {agent_status['tools_count']}")
        else:
            print("⚠️  ProductivityAgent not found in active agents")
            print(
                f"   Available agents: {list(service_info.get('agent_status', {}).keys())}"
            )

        return True

    except Exception as e:
        print(f"❌ Multi-agent service integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_productivity_routing():
    """Test routing of productivity-related queries to ProductivityAgent."""
    print("\n🧪 Testing Productivity Query Routing...")

    try:
        # Load configuration
        config_path = project_root / "src" / "voice_agent" / "config" / "default.yaml"
        config = Config.load(config_path)
        config.multi_agent.enabled = True

        # Initialize services
        from voice_agent.core.config import ToolsConfig

        tools_config = ToolsConfig(
            enabled=[
                "calculator",
                "weather",
                "file_ops",
                "calendar",
                "web_search",
                "news",
            ]
        )
        tool_executor = ToolExecutor(tools_config)

        multi_agent_service = MultiAgentService(
            config=config, tool_executor=tool_executor
        )

        await multi_agent_service.initialize()

        if not multi_agent_service.multi_agent_enabled:
            print("⚠️  Multi-agent mode not enabled, skipping routing test")
            return False

        # Test productivity-related queries
        productivity_queries = [
            "schedule a meeting for tomorrow",
            "create a calendar event",
            "read a file from my documents",
            "organize my project files",
            "create a todo list",
            "set up a reminder for next week",
        ]

        print("Testing productivity query routing:")
        for query in productivity_queries:
            try:
                response = await multi_agent_service.process_message(query)
                print(f"✅ Query: '{query[:40]}...' -> Response received")

                # Check routing stats
                service_info = multi_agent_service.get_service_info()
                routing_stats = service_info.get("routing_stats", {})
                if "productivity_agent" in routing_stats:
                    print(
                        f"   Routed to ProductivityAgent: {routing_stats['productivity_agent']} times"
                    )

            except Exception as e:
                print(f"❌ Query processing failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Productivity routing test failed: {e}")
        return False


async def test_file_operations():
    """Test file operations through ProductivityAgent."""
    print("\n🧪 Testing File Operations...")

    try:
        # Load configuration and initialize service
        config_path = project_root / "src" / "voice_agent" / "config" / "default.yaml"
        config = Config.load(config_path)
        config.multi_agent.enabled = True

        from voice_agent.core.config import ToolsConfig

        tools_config = ToolsConfig(
            enabled=[
                "calculator",
                "weather",
                "file_ops",
                "calendar",
                "web_search",
                "news",
            ]
        )
        tool_executor = ToolExecutor(tools_config)

        multi_agent_service = MultiAgentService(
            config=config, tool_executor=tool_executor
        )

        await multi_agent_service.initialize()

        if not multi_agent_service.multi_agent_enabled:
            print("⚠️  Multi-agent mode not enabled, skipping file operations test")
            return False

        # Test file-related queries
        file_queries = [
            "list files in the current directory",
            "check if a file exists at /tmp",
            "create a test file with some content",
            "read the contents of a file",
        ]

        print("Testing file operation queries:")
        for query in file_queries:
            try:
                response = await multi_agent_service.process_message(query)
                print(f"✅ File query: '{query}' -> Response: {len(response)} chars")
            except Exception as e:
                print(f"❌ File query failed: {e}")

        return True

    except Exception as e:
        print(f"❌ File operations test failed: {e}")
        return False


async def test_calendar_operations():
    """Test calendar operations through ProductivityAgent."""
    print("\n🧪 Testing Calendar Operations...")

    try:
        # Load configuration and initialize service
        config_path = project_root / "src" / "voice_agent" / "config" / "default.yaml"
        config = Config.load(config_path)
        config.multi_agent.enabled = True

        tool_registry = ToolRegistry()

        # Register built-in tools manually
        from voice_agent.tools.builtin.calculator import CalculatorTool
        from voice_agent.tools.builtin.weather import WeatherTool
        from voice_agent.tools.builtin.file_ops import FileOpsTool
        from voice_agent.tools.builtin.calendar import CalendarTool
        from voice_agent.tools.builtin.web_search import WebSearchTool
        from voice_agent.tools.builtin.news import NewsTool

        tool_registry.register_tool(CalculatorTool())
        tool_registry.register_tool(WeatherTool())
        tool_registry.register_tool(FileOpsTool())
        tool_registry.register_tool(CalendarTool())
        tool_registry.register_tool(WebSearchTool())
        tool_registry.register_tool(NewsTool())

        tool_executor = ToolExecutor(tool_registry)

        multi_agent_service = MultiAgentService(
            config=config, tool_executor=tool_executor
        )

        await multi_agent_service.initialize()

        if not multi_agent_service.multi_agent_enabled:
            print("⚠️  Multi-agent mode not enabled, skipping calendar operations test")
            return False

        # Test calendar-related queries
        calendar_queries = [
            "show my upcoming events",
            "schedule a meeting for 2pm tomorrow",
            "create a calendar appointment",
            "check my availability next week",
        ]

        print("Testing calendar operation queries:")
        for query in calendar_queries:
            try:
                response = await multi_agent_service.process_message(query)
                print(
                    f"✅ Calendar query: '{query}' -> Response: {len(response)} chars"
                )
            except Exception as e:
                print(f"❌ Calendar query failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Calendar operations test failed: {e}")
        return False


async def main():
    """Run all ProductivityAgent tests."""
    print("🚀 Starting ProductivityAgent Integration Tests")
    print("=" * 60)

    if not IMPORTS_AVAILABLE:
        print("❌ Required imports not available - skipping tests")
        return False

    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)

    # Run test suite
    test_results = []

    tests = [
        ("Agent Initialization", test_productivity_agent_initialization),
        ("Tool Integration", test_tool_integration),
        ("Multi-Agent Service Integration", test_multi_agent_service_integration),
        ("Productivity Routing", test_productivity_routing),
        ("File Operations", test_file_operations),
        ("Calendar Operations", test_calendar_operations),
    ]

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = await test_func()
            test_results.append((test_name, result))
            if result:
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
            test_results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:35} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print(
            "🎉 All tests passed! ProductivityAgent integration is working correctly."
        )
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
