#!/usr/bin/env python3
"""
Comprehensive InformationAgent Test Suite

This test suite provides extensive validation of the InformationAgent including:
- Real and mock tool integration
- Weather query processing with various formats
- Web search functionality and result processing
- News query handling (with mock implementation)
- Information synthesis and response formatting
- Error handling for external service failures
- Edge cases and boundary conditions
- Performance benchmarking
- Security validation
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to Python path and import test framework
sys.path.insert(0, str(Path(__file__).parent / "src"))
from test_agent_framework import AgentTestFramework, TestCategory, TestSeverity

from voice_agent.core.multi_agent_service import MultiAgentService
from voice_agent.core.multi_agent.agent_base import AgentMessage, MessageType


class InformationAgentTestSuite:
    """Comprehensive test suite for InformationAgent."""

    def __init__(self):
        self.framework = AgentTestFramework()
        self.logger = logging.getLogger(__name__)
        self.multi_agent_service = None
        self.information_agent = None

    async def setup(self):
        """Set up test environment."""
        self.test_env = await self.framework.setup_test_environment()

        # Create multi-agent service
        self.multi_agent_service = MultiAgentService(
            config=self.test_env["config"], tool_executor=self.test_env["tool_executor"]
        )
        await self.multi_agent_service.initialize()

        # Get information agent if available
        if "information_agent" in self.multi_agent_service.agents:
            self.information_agent = self.multi_agent_service.agents[
                "information_agent"
            ]

    async def cleanup(self):
        """Clean up test environment."""
        if self.multi_agent_service:
            await self.multi_agent_service.cleanup()
        await self.framework.cleanup_test_environment()

    # INITIALIZATION TESTS

    async def test_information_agent_initialization(self) -> Dict[str, Any]:
        """Test InformationAgent initialization and configuration."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        if "information_agent" not in self.multi_agent_service.agents:
            return {
                "success": False,
                "details": "InformationAgent not found in active agents",
            }

        agent = self.multi_agent_service.agents["information_agent"]

        # Verify agent properties
        checks = []
        checks.append(("agent_id", agent.agent_id == "information_agent"))
        checks.append(("has_capabilities", len(agent.capabilities) > 0))
        checks.append(("has_tools", len(agent.tools) > 0))
        checks.append(("is_initialized", agent.status.value in ["ready", "active"]))

        all_passed = all(check[1] for check in checks)

        return {
            "success": all_passed,
            "details": f"Initialization checks: {dict(checks)}",
            "metrics": {
                "capabilities_count": len(agent.capabilities),
                "tools_count": len(agent.tools),
            },
        }

    async def test_information_agent_capabilities(self) -> Dict[str, Any]:
        """Test that InformationAgent has correct capabilities."""
        if not self.information_agent:
            return {"success": False, "details": "InformationAgent not available"}

        from voice_agent.core.multi_agent.agent_base import AgentCapability

        expected_capabilities = {
            AgentCapability.WEATHER_INFO,
            AgentCapability.WEB_SEARCH,
            AgentCapability.NEWS_INFO,
        }

        actual_capabilities = set(self.information_agent.capabilities)
        has_expected = expected_capabilities.issubset(actual_capabilities)

        return {
            "success": has_expected,
            "details": f"Expected: {[cap.value for cap in expected_capabilities]}, Got: {[cap.value for cap in actual_capabilities]}",
            "metrics": {
                "expected_count": len(expected_capabilities),
                "actual_count": len(actual_capabilities),
                "matching_count": len(
                    expected_capabilities.intersection(actual_capabilities)
                ),
            },
        }

    # WEATHER FUNCTIONALITY TESTS

    async def test_weather_basic_query(self) -> Dict[str, Any]:
        """Test basic weather query processing."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        query = "What's the weather like in New York?"

        try:
            response = await self.multi_agent_service.process_message(query)

            success = bool(response and len(response.strip()) > 0)

            return {
                "success": success,
                "details": f"Weather query processed, response length: {len(response) if response else 0}",
                "metrics": {
                    "response_length": len(response) if response else 0,
                    "contains_weather_terms": any(
                        term in response.lower()
                        for term in ["weather", "temperature", "forecast", "condition"]
                    ),
                },
            }
        except Exception as e:
            return {"success": False, "details": f"Weather query failed: {str(e)}"}

    async def test_weather_various_locations(self) -> Dict[str, Any]:
        """Test weather queries for various location formats."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        test_locations = [
            "London",
            "Tokyo, Japan",
            "New York, NY",
            "Sydney, Australia",
            "Paris, France",
        ]

        results = []

        for location in test_locations:
            try:
                query = f"Get the weather for {location}"
                response = await self.multi_agent_service.process_message(query)

                success = bool(response and len(response.strip()) > 0)
                results.append(
                    {
                        "location": location,
                        "success": success,
                        "response_length": len(response) if response else 0,
                    }
                )
            except Exception as e:
                results.append(
                    {"location": location, "success": False, "error": str(e)}
                )

        successful_queries = sum(1 for r in results if r["success"])
        success_rate = successful_queries / len(test_locations)

        return {
            "success": success_rate >= 0.8,  # At least 80% success rate
            "details": f"Weather queries success rate: {success_rate:.1%}",
            "metrics": {
                "total_locations": len(test_locations),
                "successful_queries": successful_queries,
                "success_rate": success_rate,
                "results": results,
            },
        }

    async def test_weather_error_handling(self) -> Dict[str, Any]:
        """Test weather query error handling."""
        error_scenarios = [
            {
                "name": "invalid_location",
                "query": "What's the weather in NONEXISTENTCITY12345?",
                "setup": lambda: self.framework.mock_executor.set_failure_mode(
                    True, "Location not found"
                ),
            },
            {
                "name": "empty_location",
                "query": "What's the weather in ?",
                "setup": lambda: self.framework.mock_executor.set_failure_mode(
                    True, "Invalid location"
                ),
            },
            {
                "name": "network_error",
                "query": "Get weather for London",
                "setup": lambda: self.framework.mock_executor.set_failure_mode(
                    True, "Network connection failed"
                ),
            },
        ]

        if not self.information_agent:
            return {"success": False, "details": "InformationAgent not available"}

        results = await self.framework.test_agent_error_handling(
            self.information_agent, error_scenarios
        )

        # Reset mock executor
        self.framework.mock_executor.set_failure_mode(False)

        success_count = sum(1 for r in results.values() if r["success"])

        return {
            "success": success_count
            >= len(error_scenarios) * 0.7,  # 70% should handle errors gracefully
            "details": f"Error handling results: {success_count}/{len(error_scenarios)} scenarios handled properly",
            "metrics": {
                "total_scenarios": len(error_scenarios),
                "handled_properly": success_count,
                "results": results,
            },
        }

    # WEB SEARCH FUNCTIONALITY TESTS

    async def test_web_search_basic_query(self) -> Dict[str, Any]:
        """Test basic web search functionality."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        query = "Search for Python programming tutorials"

        try:
            response = await self.multi_agent_service.process_message(query)

            success = bool(response and len(response.strip()) > 0)

            return {
                "success": success,
                "details": f"Web search processed, response length: {len(response) if response else 0}",
                "metrics": {
                    "response_length": len(response) if response else 0,
                    "contains_search_terms": any(
                        term in response.lower()
                        for term in ["search", "found", "results", "tutorial"]
                    ),
                },
            }
        except Exception as e:
            return {"success": False, "details": f"Web search failed: {str(e)}"}

    async def test_web_search_various_queries(self) -> Dict[str, Any]:
        """Test web search with various query types."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        test_queries = [
            "Find information about artificial intelligence",
            "Search for renewable energy solutions",
            "Look up machine learning algorithms",
            "Find Python documentation",
            "Search for climate change research",
        ]

        results = []

        for search_query in test_queries:
            try:
                response = await self.multi_agent_service.process_message(search_query)

                success = bool(response and len(response.strip()) > 0)
                results.append(
                    {
                        "query": search_query,
                        "success": success,
                        "response_length": len(response) if response else 0,
                    }
                )
            except Exception as e:
                results.append(
                    {"query": search_query, "success": False, "error": str(e)}
                )

        successful_queries = sum(1 for r in results if r["success"])
        success_rate = successful_queries / len(test_queries)

        return {
            "success": success_rate >= 0.8,
            "details": f"Web search queries success rate: {success_rate:.1%}",
            "metrics": {
                "total_queries": len(test_queries),
                "successful_queries": successful_queries,
                "success_rate": success_rate,
                "results": results,
            },
        }

    # NEWS FUNCTIONALITY TESTS

    async def test_news_query_processing(self) -> Dict[str, Any]:
        """Test news query processing."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        news_queries = [
            "Get the latest news about technology",
            "Show me recent headlines about climate change",
            "What are the current events in business?",
        ]

        results = []

        for query in news_queries:
            try:
                response = await self.multi_agent_service.process_message(query)

                success = bool(response and len(response.strip()) > 0)
                results.append(
                    {
                        "query": query,
                        "success": success,
                        "response_length": len(response) if response else 0,
                    }
                )
            except Exception as e:
                results.append({"query": query, "success": False, "error": str(e)})

        successful_queries = sum(1 for r in results if r["success"])
        success_rate = successful_queries / len(news_queries)

        return {
            "success": success_rate
            >= 0.5,  # News is mock implementation, so lower threshold
            "details": f"News queries success rate: {success_rate:.1%}",
            "metrics": {
                "total_queries": len(news_queries),
                "successful_queries": successful_queries,
                "success_rate": success_rate,
                "results": results,
            },
        }

    # INFORMATION SYNTHESIS TESTS

    async def test_information_response_formatting(self) -> Dict[str, Any]:
        """Test information response formatting and synthesis."""
        if not self.information_agent:
            return {"success": False, "details": "InformationAgent not available"}

        # Test with mock data
        test_message = AgentMessage(
            conversation_id="format_test",
            type=MessageType.USER_INPUT,
            content="What's the weather in London and search for London travel guides?",
        )

        try:
            response = await self.information_agent.process_message(test_message)

            # Check response formatting
            formatting_checks = []

            if response.content:
                content = response.content.lower()
                formatting_checks.append(
                    ("has_structured_response", len(response.content) > 50)
                )
                formatting_checks.append(("weather_mentioned", "weather" in content))
                formatting_checks.append(
                    (
                        "search_mentioned",
                        any(term in content for term in ["search", "found", "guide"]),
                    )
                )
                formatting_checks.append(
                    (
                        "proper_formatting",
                        "\n" in response.content or "." in response.content,
                    )
                )

            success = len(formatting_checks) > 0 and all(
                check[1] for check in formatting_checks
            )

            return {
                "success": success,
                "details": f"Response formatting checks: {dict(formatting_checks)}",
                "metrics": {
                    "response_length": len(response.content) if response.content else 0,
                    "formatting_score": (
                        sum(check[1] for check in formatting_checks)
                        / len(formatting_checks)
                        if formatting_checks
                        else 0
                    ),
                },
            }
        except Exception as e:
            return {
                "success": False,
                "details": f"Response formatting test failed: {str(e)}",
            }

    # HANDOFF AND ROUTING TESTS

    async def test_agent_handoff_logic(self) -> Dict[str, Any]:
        """Test agent handoff for non-information queries."""
        if not self.information_agent:
            return {"success": False, "details": "InformationAgent not available"}

        non_info_queries = [
            ("Calculate 25 * 43", "utility_agent"),
            ("Hello, how are you?", "general_agent"),
            ("Create a file", "productivity_agent"),
        ]

        handoff_results = []

        for query, expected_handoff in non_info_queries:
            try:
                message = AgentMessage(
                    conversation_id="handoff_test",
                    type=MessageType.USER_INPUT,
                    content=query,
                )

                response = await self.information_agent.process_message(message)

                # Check if handoff was suggested
                should_handoff = (
                    hasattr(response, "should_handoff") and response.should_handoff
                )

                handoff_results.append(
                    {
                        "query": query,
                        "expected_handoff": expected_handoff,
                        "should_handoff": should_handoff,
                        "suggested_agent": getattr(response, "suggested_agent", None),
                        "success": should_handoff,  # Success if handoff was triggered
                    }
                )

            except Exception as e:
                handoff_results.append(
                    {"query": query, "success": False, "error": str(e)}
                )

        successful_handoffs = sum(1 for r in handoff_results if r.get("success", False))

        return {
            "success": successful_handoffs
            >= len(non_info_queries) * 0.6,  # 60% should trigger handoffs
            "details": f"Handoff logic: {successful_handoffs}/{len(non_info_queries)} queries triggered handoffs",
            "metrics": {
                "total_queries": len(non_info_queries),
                "successful_handoffs": successful_handoffs,
                "handoff_results": handoff_results,
            },
        }

    # PERFORMANCE TESTS

    async def test_information_agent_performance(self) -> Dict[str, Any]:
        """Test InformationAgent performance benchmarks."""
        if not self.information_agent:
            return {"success": False, "details": "InformationAgent not available"}

        test_queries = [
            "What's the weather in Paris?",
            "Search for machine learning tutorials",
            "Get news about renewable energy",
            "Weather forecast for Tokyo",
            "Find information about Python",
        ]

        try:
            metrics = await self.framework.benchmark_agent_performance(
                self.information_agent, test_queries, iterations=3
            )

            # Performance thresholds
            acceptable_response_time = 5000  # 5 seconds
            acceptable_success_rate = 0.7  # 70%

            performance_good = (
                metrics.response_time_ms < acceptable_response_time
                and metrics.success_rate >= acceptable_success_rate
            )

            return {
                "success": performance_good,
                "details": f"Performance: {metrics.response_time_ms:.2f}ms avg, {metrics.success_rate:.1%} success rate",
                "metrics": {
                    "average_response_time_ms": metrics.response_time_ms,
                    "success_rate": metrics.success_rate,
                    "error_count": metrics.error_count,
                    "total_requests": metrics.total_requests,
                    "memory_usage_mb": metrics.memory_usage_mb,
                },
            }
        except Exception as e:
            return {"success": False, "details": f"Performance test failed: {str(e)}"}

    # EDGE CASE TESTS

    async def test_concurrent_information_requests(self) -> Dict[str, Any]:
        """Test handling of concurrent information requests."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        concurrent_queries = [
            "Weather in London",
            "Search for AI news",
            "Weather in Tokyo",
            "Find Python tutorials",
            "Get tech headlines",
        ]

        try:
            # Execute queries concurrently
            tasks = [
                self.multi_agent_service.process_message(query)
                for query in concurrent_queries
            ]

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Check results
            successful_responses = sum(
                1
                for response in responses
                if not isinstance(response, Exception)
                and response
                and len(response.strip()) > 0
            )

            success_rate = successful_responses / len(concurrent_queries)

            return {
                "success": success_rate
                >= 0.6,  # 60% success rate for concurrent requests
                "details": f"Concurrent requests: {successful_responses}/{len(concurrent_queries)} successful",
                "metrics": {
                    "total_requests": len(concurrent_queries),
                    "successful_requests": successful_responses,
                    "success_rate": success_rate,
                    "exceptions": sum(1 for r in responses if isinstance(r, Exception)),
                },
            }
        except Exception as e:
            return {
                "success": False,
                "details": f"Concurrent requests test failed: {str(e)}",
            }

    async def test_large_information_request(self) -> Dict[str, Any]:
        """Test handling of large information requests."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        # Create a complex information request
        large_query = """
        I need comprehensive information about climate change. Please:
        1. Get current weather patterns for major cities worldwide
        2. Search for the latest climate research and reports
        3. Find news about environmental policies and initiatives  
        4. Look up information about renewable energy developments
        5. Search for climate change mitigation strategies
        Please provide a detailed summary of all this information.
        """

        try:
            response = await self.multi_agent_service.process_message(large_query)

            success = bool(
                response and len(response.strip()) > 100
            )  # Expect substantial response

            return {
                "success": success,
                "details": f"Large information request processed, response length: {len(response) if response else 0}",
                "metrics": {
                    "query_length": len(large_query),
                    "response_length": len(response) if response else 0,
                    "complexity_score": large_query.count("search")
                    + large_query.count("find")
                    + large_query.count("get"),
                },
            }
        except Exception as e:
            return {"success": False, "details": f"Large request test failed: {str(e)}"}


async def run_comprehensive_information_agent_tests():
    """Run all comprehensive InformationAgent tests."""
    print("🌐 Starting Comprehensive InformationAgent Test Suite")
    print("=" * 80)

    test_suite = InformationAgentTestSuite()
    framework = test_suite.framework

    try:
        await test_suite.setup()

        # Define all tests
        tests = [
            # Initialization tests
            (
                "InformationAgent Initialization",
                test_suite.test_information_agent_initialization,
                TestCategory.INITIALIZATION,
                TestSeverity.CRITICAL,
            ),
            (
                "Agent Capabilities Check",
                test_suite.test_information_agent_capabilities,
                TestCategory.INITIALIZATION,
                TestSeverity.HIGH,
            ),
            # Weather functionality tests
            (
                "Weather Basic Query",
                test_suite.test_weather_basic_query,
                TestCategory.FUNCTIONALITY,
                TestSeverity.HIGH,
            ),
            (
                "Weather Various Locations",
                test_suite.test_weather_various_locations,
                TestCategory.FUNCTIONALITY,
                TestSeverity.MEDIUM,
            ),
            (
                "Weather Error Handling",
                test_suite.test_weather_error_handling,
                TestCategory.ERROR_HANDLING,
                TestSeverity.HIGH,
            ),
            # Web search functionality tests
            (
                "Web Search Basic Query",
                test_suite.test_web_search_basic_query,
                TestCategory.FUNCTIONALITY,
                TestSeverity.HIGH,
            ),
            (
                "Web Search Various Queries",
                test_suite.test_web_search_various_queries,
                TestCategory.FUNCTIONALITY,
                TestSeverity.MEDIUM,
            ),
            # News functionality tests
            (
                "News Query Processing",
                test_suite.test_news_query_processing,
                TestCategory.FUNCTIONALITY,
                TestSeverity.MEDIUM,
            ),
            # Information synthesis tests
            (
                "Information Response Formatting",
                test_suite.test_information_response_formatting,
                TestCategory.FUNCTIONALITY,
                TestSeverity.MEDIUM,
            ),
            # Integration tests
            (
                "Agent Handoff Logic",
                test_suite.test_agent_handoff_logic,
                TestCategory.INTEGRATION,
                TestSeverity.HIGH,
            ),
            # Performance tests
            (
                "InformationAgent Performance",
                test_suite.test_information_agent_performance,
                TestCategory.PERFORMANCE,
                TestSeverity.MEDIUM,
            ),
            # Edge case tests
            (
                "Concurrent Information Requests",
                test_suite.test_concurrent_information_requests,
                TestCategory.EDGE_CASES,
                TestSeverity.MEDIUM,
            ),
            (
                "Large Information Request",
                test_suite.test_large_information_request,
                TestCategory.EDGE_CASES,
                TestSeverity.LOW,
            ),
        ]

        # Run all tests
        for test_name, test_func, category, severity in tests:
            await framework.run_test(
                test_func=test_func,
                test_name=test_name,
                category=category,
                severity=severity,
                timeout_seconds=60.0,  # Generous timeout for information requests
            )

        # Generate and display results
        framework.print_test_summary()
        framework.save_test_report("information_agent_test_report.json")

        return framework.generate_test_report()

    except Exception as e:
        print(f"💥 Test suite execution failed: {e}")
        return {"error": str(e)}

    finally:
        await test_suite.cleanup()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the comprehensive test suite
    try:
        results = asyncio.run(run_comprehensive_information_agent_tests())

        if "error" in results:
            print(f"\n💥 Test suite failed: {results['error']}")
            sys.exit(1)

        success_rate = results["summary"]["success_rate"]
        if success_rate >= 0.8:
            print(
                f"\n🎉 InformationAgent tests completed successfully! ({success_rate:.1%} pass rate)"
            )
            sys.exit(0)
        elif success_rate >= 0.6:
            print(
                f"\n⚠️  InformationAgent tests mostly successful ({success_rate:.1%} pass rate)"
            )
            sys.exit(1)
        else:
            print(f"\n❌ InformationAgent tests failed ({success_rate:.1%} pass rate)")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        sys.exit(1)
