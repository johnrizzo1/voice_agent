#!/usr/bin/env python3
"""
Comprehensive Agent Integration Test Suite

This test suite provides extensive validation of agent integration including:
- Agent routing and delegation decisions based on query types
- Context sharing between agents during handoffs
- Multi-agent workflow coordination and collaboration
- Load balancing among capable agents
- Agent communication patterns and message passing
- Fallback mechanisms when agents are unavailable
- Cross-agent context preservation and memory
- Performance of multi-agent orchestration
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


class AgentIntegrationTestSuite:
    """Comprehensive test suite for agent integration and routing."""

    def __init__(self):
        self.framework = AgentTestFramework()
        self.logger = logging.getLogger(__name__)
        self.multi_agent_service = None

    async def setup(self):
        """Set up test environment."""
        self.test_env = await self.framework.setup_test_environment()

        # Create multi-agent service
        self.multi_agent_service = MultiAgentService(
            config=self.test_env["config"], tool_executor=self.test_env["tool_executor"]
        )
        await self.multi_agent_service.initialize()

    async def cleanup(self):
        """Clean up test environment."""
        if self.multi_agent_service:
            await self.multi_agent_service.cleanup()
        await self.framework.cleanup_test_environment()

    # ROUTING AND DELEGATION TESTS

    async def test_query_routing_accuracy(self) -> Dict[str, Any]:
        """Test accuracy of query routing to appropriate agents."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        # Define queries with expected agent assignments
        routing_test_cases = [
            # Information queries -> InformationAgent
            ("What's the weather in London?", "information"),
            ("Search for Python tutorials", "information"),
            ("Get the latest tech news", "information"),
            # Mathematical queries -> UtilityAgent
            ("Calculate 25 * 43 + 17", "utility"),
            ("What is 15% of 200?", "utility"),
            ("Solve the equation 2x + 5 = 15", "utility"),
            # Productivity queries -> ProductivityAgent
            ("Create a calendar event for tomorrow", "productivity"),
            ("List files in my documents", "productivity"),
            ("Schedule a team meeting", "productivity"),
            # General queries -> GeneralAgent (fallback patterns)
            ("Hello, how are you?", "general"),
            ("Tell me a joke", "general"),
            ("Help me understand this concept", "general"),
        ]

        routing_results = []

        for query, expected_agent_type in routing_test_cases:
            try:
                # Clear previous routing stats
                if hasattr(self.multi_agent_service, "_routing_stats"):
                    self.multi_agent_service._routing_stats.clear()

                response = await self.multi_agent_service.process_message(query)

                # Check service info for routing stats
                service_info = self.multi_agent_service.get_service_info()
                routing_stats = service_info.get("routing_stats", {})

                # Determine which agent handled the query
                routed_agent = None
                for agent_name, count in routing_stats.items():
                    if count > 0:
                        routed_agent = agent_name
                        break

                # Check if routing was correct
                correct_routing = (
                    routed_agent and expected_agent_type in routed_agent.lower()
                )

                routing_results.append(
                    {
                        "query": query,
                        "expected_agent": expected_agent_type,
                        "routed_agent": routed_agent,
                        "correct_routing": correct_routing,
                        "response": response,
                        "success": bool(response and len(response.strip()) > 0),
                    }
                )

            except Exception as e:
                routing_results.append(
                    {
                        "query": query,
                        "expected_agent": expected_agent_type,
                        "success": False,
                        "error": str(e),
                    }
                )

        correct_routings = sum(
            1 for r in routing_results if r.get("correct_routing", False)
        )
        successful_responses = sum(1 for r in routing_results if r["success"])
        routing_accuracy = correct_routings / len(routing_test_cases)
        response_success_rate = successful_responses / len(routing_test_cases)

        return {
            "success": routing_accuracy >= 0.7 and response_success_rate >= 0.8,
            "details": f"Routing accuracy: {routing_accuracy:.1%}, Response success: {response_success_rate:.1%}",
            "metrics": {
                "total_queries": len(routing_test_cases),
                "correct_routings": correct_routings,
                "successful_responses": successful_responses,
                "routing_accuracy": routing_accuracy,
                "response_success_rate": response_success_rate,
                "routing_results": routing_results,
            },
        }

    async def test_agent_handoff_mechanisms(self) -> Dict[str, Any]:
        """Test agent handoff mechanisms for complex queries."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        # Complex queries that might require handoffs between agents
        handoff_test_cases = [
            "What's the weather in Paris and calculate the temperature difference from 20°C?",
            "Search for file organization tips and help me create a folder structure",
            "Find information about meeting scheduling best practices and set up a calendar event",
            "Calculate the monthly budget (income $5000, expenses $3500) and create a financial summary file",
        ]

        handoff_results = []

        for query in handoff_test_cases:
            try:
                conversation_id = f"handoff_test_{len(handoff_results)}"

                response = await self.multi_agent_service.process_message(
                    query, conversation_id=conversation_id
                )

                # Check if response addresses multiple aspects of the query
                response_lower = response.lower() if response else ""
                addresses_multiple_aspects = (
                    len(
                        [
                            word
                            for word in [
                                "weather",
                                "calculate",
                                "search",
                                "file",
                                "calendar",
                                "budget",
                            ]
                            if word in response_lower
                        ]
                    )
                    >= 2
                )

                # Check service info for agent switches
                service_info = self.multi_agent_service.get_service_info()
                agent_switches = service_info.get("agent_switches", 0)

                handoff_results.append(
                    {
                        "query": query,
                        "response": response,
                        "addresses_multiple_aspects": addresses_multiple_aspects,
                        "agent_switches": agent_switches,
                        "success": bool(response and len(response.strip()) > 0),
                    }
                )

            except Exception as e:
                handoff_results.append(
                    {"query": query, "success": False, "error": str(e)}
                )

        successful_handoffs = sum(1 for r in handoff_results if r["success"])
        multi_aspect_responses = sum(
            1 for r in handoff_results if r.get("addresses_multiple_aspects", False)
        )

        return {
            "success": successful_handoffs >= len(handoff_test_cases) * 0.7,
            "details": f"Handoff mechanisms: {successful_handoffs}/{len(handoff_test_cases)} successful, {multi_aspect_responses} multi-aspect responses",
            "metrics": {
                "total_queries": len(handoff_test_cases),
                "successful_handoffs": successful_handoffs,
                "multi_aspect_responses": multi_aspect_responses,
                "handoff_results": handoff_results,
            },
        }

    async def test_load_balancing_among_agents(self) -> Dict[str, Any]:
        """Test load balancing when multiple agents can handle similar queries."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        # Queries that could potentially be handled by multiple agents
        ambiguous_queries = [
            "Help me with calculations and file management",
            "I need information and also want to organize my tasks",
            "Can you search for data and help me schedule time to review it?",
            "Find weather information and save it to a file",
        ] * 3  # Repeat to test load balancing

        agent_usage_count = {}
        successful_queries = 0

        for i, query in enumerate(ambiguous_queries):
            try:
                conversation_id = f"load_balance_test_{i}"
                response = await self.multi_agent_service.process_message(
                    query, conversation_id=conversation_id
                )

                if response and len(response.strip()) > 0:
                    successful_queries += 1

                # Track which agents were used
                service_info = self.multi_agent_service.get_service_info()
                routing_stats = service_info.get("routing_stats", {})

                for agent_name, count in routing_stats.items():
                    if count > 0:
                        agent_usage_count[agent_name] = (
                            agent_usage_count.get(agent_name, 0) + 1
                        )

            except Exception as e:
                self.logger.warning(f"Load balancing test query failed: {e}")

        # Check if load is reasonably distributed (no single agent handles everything)
        total_agent_uses = sum(agent_usage_count.values())
        if total_agent_uses > 0:
            max_agent_usage = (
                max(agent_usage_count.values()) if agent_usage_count else 0
            )
            load_balance_score = (
                1.0 - (max_agent_usage / total_agent_uses)
                if total_agent_uses > 0
                else 0.0
            )
        else:
            load_balance_score = 0.0

        success_rate = successful_queries / len(ambiguous_queries)

        return {
            "success": success_rate >= 0.7
            and load_balance_score >= 0.3,  # Some distribution expected
            "details": f"Load balancing: {success_rate:.1%} success rate, {load_balance_score:.2f} balance score",
            "metrics": {
                "total_queries": len(ambiguous_queries),
                "successful_queries": successful_queries,
                "success_rate": success_rate,
                "agent_usage_count": agent_usage_count,
                "load_balance_score": load_balance_score,
                "unique_agents_used": len(agent_usage_count),
            },
        }

    # CONTEXT PRESERVATION TESTS

    async def test_cross_agent_context_preservation(self) -> Dict[str, Any]:
        """Test context preservation when queries are handed off between agents."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        conversation_id = "cross_agent_context_test"

        try:
            # Multi-step conversation that should involve different agents
            responses = []

            # Step 1: Information query (should go to InformationAgent)
            response1 = await self.multi_agent_service.process_message(
                "What's the current weather in New York?",
                conversation_id=conversation_id,
            )
            responses.append(("weather_query", response1))

            # Step 2: Mathematical query referencing previous context (should go to UtilityAgent)
            response2 = await self.multi_agent_service.process_message(
                "If the temperature you just mentioned is in Fahrenheit, convert it to Celsius",
                conversation_id=conversation_id,
            )
            responses.append(("temperature_conversion", response2))

            # Step 3: Productivity query building on context (should go to ProductivityAgent)
            response3 = await self.multi_agent_service.process_message(
                "Create a weather report file with that temperature information",
                conversation_id=conversation_id,
            )
            responses.append(("file_creation", response3))

            # Step 4: General query that references the whole conversation
            response4 = await self.multi_agent_service.process_message(
                "Summarize what we've accomplished in this conversation",
                conversation_id=conversation_id,
            )
            responses.append(("summary", response4))

            # Analyze context preservation
            all_responses_received = all(
                r[1] and len(r[1].strip()) > 0 for r in responses
            )

            # Check if later responses reference earlier context
            context_references = 0
            if response2 and (
                "temperature" in response2.lower()
                or "fahrenheit" in response2.lower()
                or "celsius" in response2.lower()
            ):
                context_references += 1
            if response3 and (
                "weather" in response3.lower()
                or "temperature" in response3.lower()
                or "file" in response3.lower()
            ):
                context_references += 1
            if response4 and (
                "weather" in response4.lower()
                or "temperature" in response4.lower()
                or "conversation" in response4.lower()
            ):
                context_references += 1

            context_preservation_score = (
                context_references / 3
            )  # Out of 3 possible references

            return {
                "success": all_responses_received and context_preservation_score >= 0.6,
                "details": f"Cross-agent context: {context_preservation_score:.1%} preservation, all responses: {all_responses_received}",
                "metrics": {
                    "total_steps": len(responses),
                    "all_responses_received": all_responses_received,
                    "context_references": context_references,
                    "context_preservation_score": context_preservation_score,
                    "response_lengths": [len(r[1]) if r[1] else 0 for r in responses],
                },
            }

        except Exception as e:
            return {
                "success": False,
                "details": f"Cross-agent context test failed: {str(e)}",
            }

    async def test_agent_memory_consistency(self) -> Dict[str, Any]:
        """Test memory consistency across different agents in the same conversation."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        conversation_id = "memory_consistency_test"

        try:
            # Establish some facts that should be remembered
            setup_queries = [
                "My name is Alex and I work as a software engineer",
                "I'm working on a project called 'DataViz Pro'",
                "The project deadline is next Friday",
            ]

            for query in setup_queries:
                await self.multi_agent_service.process_message(
                    query, conversation_id=conversation_id
                )

            # Test memory with different types of queries (different agents)
            memory_test_queries = [
                ("Calculate how many days until next Friday", "utility"),
                (
                    "Search for information about data visualization tools",
                    "information",
                ),
                (
                    "Create a calendar reminder for the DataViz Pro deadline",
                    "productivity",
                ),
                ("What did I tell you about my project?", "general"),
            ]

            memory_results = []

            for query, expected_agent_type in memory_test_queries:
                response = await self.multi_agent_service.process_message(
                    query, conversation_id=conversation_id
                )

                # Check if response shows memory of established facts
                response_lower = response.lower() if response else ""
                remembers_facts = any(
                    fact in response_lower
                    for fact in [
                        "alex",
                        "software engineer",
                        "dataviz",
                        "project",
                        "deadline",
                        "friday",
                    ]
                )

                memory_results.append(
                    {
                        "query": query,
                        "expected_agent": expected_agent_type,
                        "response": response,
                        "remembers_facts": remembers_facts,
                        "success": bool(response and len(response.strip()) > 0),
                    }
                )

            successful_memory_tests = sum(
                1 for r in memory_results if r["remembers_facts"]
            )
            total_successful_responses = sum(1 for r in memory_results if r["success"])

            memory_consistency_score = successful_memory_tests / len(
                memory_test_queries
            )
            response_success_rate = total_successful_responses / len(
                memory_test_queries
            )

            return {
                "success": memory_consistency_score >= 0.5
                and response_success_rate >= 0.7,
                "details": f"Memory consistency: {memory_consistency_score:.1%}, Response success: {response_success_rate:.1%}",
                "metrics": {
                    "total_queries": len(memory_test_queries),
                    "successful_memory_tests": successful_memory_tests,
                    "memory_consistency_score": memory_consistency_score,
                    "response_success_rate": response_success_rate,
                    "memory_results": memory_results,
                },
            }

        except Exception as e:
            return {
                "success": False,
                "details": f"Agent memory consistency test failed: {str(e)}",
            }

    # COLLABORATION AND WORKFLOW TESTS

    async def test_multi_agent_collaborative_workflows(self) -> Dict[str, Any]:
        """Test multi-agent collaborative workflows."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        # Complex workflows that require multiple agents working together
        collaborative_workflows = [
            {
                "name": "research_and_document",
                "description": "Research a topic and create documentation",
                "query": "Research the benefits of remote work and create a summary document",
            },
            {
                "name": "data_analysis_workflow",
                "description": "Analyze data and schedule review meeting",
                "query": "Calculate the average of these numbers: 15, 23, 31, 8, 19, and schedule a data review meeting",
            },
            {
                "name": "project_planning_workflow",
                "description": "Search for project management tips and create a task list",
                "query": "Find information about agile project management and help me create a sprint planning checklist",
            },
        ]

        workflow_results = []

        for workflow in collaborative_workflows:
            try:
                conversation_id = f"workflow_{workflow['name']}"

                response = await self.multi_agent_service.process_message(
                    workflow["query"], conversation_id=conversation_id
                )

                # Check if response addresses multiple aspects of the workflow
                response_lower = response.lower() if response else ""

                # Count workflow indicators
                workflow_indicators = 0
                if "research" in workflow["query"].lower() and any(
                    term in response_lower
                    for term in ["research", "information", "found"]
                ):
                    workflow_indicators += 1
                if "calculate" in workflow["query"].lower() and any(
                    term in response_lower
                    for term in ["calculate", "average", "result"]
                ):
                    workflow_indicators += 1
                if "create" in workflow["query"].lower() and any(
                    term in response_lower
                    for term in ["create", "document", "file", "list"]
                ):
                    workflow_indicators += 1
                if "schedule" in workflow["query"].lower() and any(
                    term in response_lower
                    for term in ["schedule", "meeting", "calendar"]
                ):
                    workflow_indicators += 1

                collaborative_success = workflow_indicators >= 1 and bool(
                    response and len(response.strip()) > 100
                )

                workflow_results.append(
                    {
                        "workflow": workflow["name"],
                        "query": workflow["query"],
                        "response": response,
                        "workflow_indicators": workflow_indicators,
                        "collaborative_success": collaborative_success,
                        "success": bool(response and len(response.strip()) > 0),
                    }
                )

            except Exception as e:
                workflow_results.append(
                    {
                        "workflow": workflow["name"],
                        "query": workflow["query"],
                        "success": False,
                        "error": str(e),
                    }
                )

        successful_workflows = sum(
            1 for r in workflow_results if r.get("collaborative_success", False)
        )
        total_successful_responses = sum(1 for r in workflow_results if r["success"])

        return {
            "success": successful_workflows >= len(collaborative_workflows) * 0.6,
            "details": f"Collaborative workflows: {successful_workflows}/{len(collaborative_workflows)} successful",
            "metrics": {
                "total_workflows": len(collaborative_workflows),
                "successful_workflows": successful_workflows,
                "total_successful_responses": total_successful_responses,
                "workflow_results": workflow_results,
            },
        }

    # FALLBACK AND ERROR HANDLING TESTS

    async def test_agent_fallback_mechanisms(self) -> Dict[str, Any]:
        """Test fallback mechanisms when specific agents are unavailable."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        # Simulate different fallback scenarios
        fallback_test_cases = [
            (
                "What's the weather today?",
                "Should fallback gracefully if weather service unavailable",
            ),
            (
                "Calculate complex mathematical expression",
                "Should fallback to basic calculation if advanced math unavailable",
            ),
            (
                "Create a detailed project plan",
                "Should fallback to general planning advice if specific tools unavailable",
            ),
        ]

        fallback_results = []

        for query, expected_behavior in fallback_test_cases:
            try:
                # Test with potential service degradation
                self.framework.mock_executor.set_failure_mode(
                    True, "Service temporarily unavailable"
                )

                response = await self.multi_agent_service.process_message(query)

                # Reset failure mode
                self.framework.mock_executor.set_failure_mode(False)

                # Check if system handled the fallback gracefully
                handles_fallback = bool(response and len(response.strip()) > 0)

                # Look for fallback indicators
                response_lower = response.lower() if response else ""
                shows_degraded_service = any(
                    term in response_lower
                    for term in [
                        "sorry",
                        "unable",
                        "limited",
                        "try",
                        "alternative",
                        "currently",
                    ]
                )

                fallback_results.append(
                    {
                        "query": query,
                        "expected_behavior": expected_behavior,
                        "response": response,
                        "handles_fallback": handles_fallback,
                        "shows_degraded_service": shows_degraded_service,
                        "success": handles_fallback,
                    }
                )

            except Exception as e:
                # Reset failure mode in case of exception
                self.framework.mock_executor.set_failure_mode(False)

                fallback_results.append(
                    {
                        "query": query,
                        "expected_behavior": expected_behavior,
                        "success": False,
                        "error": str(e),
                    }
                )

        successful_fallbacks = sum(1 for r in fallback_results if r["success"])

        return {
            "success": successful_fallbacks >= len(fallback_test_cases) * 0.8,
            "details": f"Fallback mechanisms: {successful_fallbacks}/{len(fallback_test_cases)} handled gracefully",
            "metrics": {
                "total_fallback_tests": len(fallback_test_cases),
                "successful_fallbacks": successful_fallbacks,
                "fallback_results": fallback_results,
            },
        }

    # PERFORMANCE TESTS

    async def test_multi_agent_orchestration_performance(self) -> Dict[str, Any]:
        """Test performance of multi-agent orchestration."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        # Performance test queries that exercise different agents
        performance_queries = [
            "What's the weather and calculate the temperature in Celsius?",
            "Search for project management tips and create a task list",
            "Find information about Python and schedule time to study it",
            "Calculate my monthly savings and create a budget file",
        ]

        try:
            # Run performance benchmark
            import time

            start_time = time.time()
            successful_queries = 0
            total_response_length = 0

            for query in performance_queries:
                query_start = time.time()

                try:
                    response = await self.multi_agent_service.process_message(query)

                    if response and len(response.strip()) > 0:
                        successful_queries += 1
                        total_response_length += len(response)

                    query_duration = (time.time() - query_start) * 1000

                    # Log if query took too long
                    if query_duration > 10000:  # 10 seconds
                        self.logger.warning(
                            f"Slow multi-agent query: {query_duration:.2f}ms"
                        )

                except Exception as e:
                    self.logger.error(f"Performance test query failed: {e}")

            total_duration = (time.time() - start_time) * 1000
            average_duration = total_duration / len(performance_queries)
            success_rate = successful_queries / len(performance_queries)

            # Performance thresholds
            acceptable_avg_duration = 8000  # 8 seconds per query
            acceptable_success_rate = 0.7  # 70% success rate

            performance_acceptable = (
                average_duration < acceptable_avg_duration
                and success_rate >= acceptable_success_rate
            )

            return {
                "success": performance_acceptable,
                "details": f"Multi-agent performance: {average_duration:.2f}ms avg, {success_rate:.1%} success rate",
                "metrics": {
                    "total_queries": len(performance_queries),
                    "successful_queries": successful_queries,
                    "total_duration_ms": total_duration,
                    "average_duration_ms": average_duration,
                    "success_rate": success_rate,
                    "total_response_length": total_response_length,
                    "performance_acceptable": performance_acceptable,
                },
            }

        except Exception as e:
            return {
                "success": False,
                "details": f"Multi-agent performance test failed: {str(e)}",
            }


async def run_comprehensive_agent_integration_tests():
    """Run all comprehensive agent integration tests."""
    print("🔗 Starting Comprehensive Agent Integration Test Suite")
    print("=" * 80)

    test_suite = AgentIntegrationTestSuite()
    framework = test_suite.framework

    try:
        await test_suite.setup()

        # Define all tests
        tests = [
            # Routing and delegation tests
            (
                "Query Routing Accuracy",
                test_suite.test_query_routing_accuracy,
                TestCategory.INTEGRATION,
                TestSeverity.CRITICAL,
            ),
            (
                "Agent Handoff Mechanisms",
                test_suite.test_agent_handoff_mechanisms,
                TestCategory.INTEGRATION,
                TestSeverity.HIGH,
            ),
            (
                "Load Balancing Among Agents",
                test_suite.test_load_balancing_among_agents,
                TestCategory.INTEGRATION,
                TestSeverity.MEDIUM,
            ),
            # Context preservation tests
            (
                "Cross-Agent Context Preservation",
                test_suite.test_cross_agent_context_preservation,
                TestCategory.INTEGRATION,
                TestSeverity.HIGH,
            ),
            (
                "Agent Memory Consistency",
                test_suite.test_agent_memory_consistency,
                TestCategory.INTEGRATION,
                TestSeverity.MEDIUM,
            ),
            # Collaboration and workflow tests
            (
                "Multi-Agent Collaborative Workflows",
                test_suite.test_multi_agent_collaborative_workflows,
                TestCategory.FUNCTIONALITY,
                TestSeverity.MEDIUM,
            ),
            # Fallback and error handling tests
            (
                "Agent Fallback Mechanisms",
                test_suite.test_agent_fallback_mechanisms,
                TestCategory.ERROR_HANDLING,
                TestSeverity.HIGH,
            ),
            # Performance tests
            (
                "Multi-Agent Orchestration Performance",
                test_suite.test_multi_agent_orchestration_performance,
                TestCategory.PERFORMANCE,
                TestSeverity.MEDIUM,
            ),
        ]

        # Run all tests
        for test_name, test_func, category, severity in tests:
            await framework.run_test(
                test_func=test_func,
                test_name=test_name,
                category=category,
                severity=severity,
                timeout_seconds=60.0,  # Generous timeout for multi-agent operations
            )

        # Generate and display results
        framework.print_test_summary()
        framework.save_test_report("agent_integration_test_report.json")

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
        results = asyncio.run(run_comprehensive_agent_integration_tests())

        if "error" in results:
            print(f"\n💥 Test suite failed: {results['error']}")
            sys.exit(1)

        success_rate = results["summary"]["success_rate"]
        if success_rate >= 0.8:
            print(
                f"\n🎉 Agent integration tests completed successfully! ({success_rate:.1%} pass rate)"
            )
            sys.exit(0)
        elif success_rate >= 0.6:
            print(
                f"\n⚠️  Agent integration tests mostly successful ({success_rate:.1%} pass rate)"
            )
            sys.exit(1)
        else:
            print(f"\n❌ Agent integration tests failed ({success_rate:.1%} pass rate)")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        sys.exit(1)
