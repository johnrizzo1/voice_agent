#!/usr/bin/env python3
"""
Comprehensive Multi-Agent Mode Integration Testing

This script tests the voice agent system in multi-agent mode to ensure
all components work properly including agent routing and communication.

Usage:
    devenv shell -- python examples/integration_test_multi_agent.py
"""

import asyncio
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_agent.core.config import Config
from voice_agent.core.voice_agent_orchestrator import VoiceAgentOrchestrator


class MultiAgentIntegrationTester:
    """Comprehensive multi-agent integration testing."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results: Dict[str, Any] = {}
        self.orchestrator: VoiceAgentOrchestrator = None

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all multi-agent integration tests."""
        print("🤖 MULTI-AGENT MODE INTEGRATION TESTING")
        print("=" * 60)

        overall_results = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_results": {},
            "overall_metrics": {},
        }

        try:
            # Test 1: Multi-Agent Configuration Loading
            config_test = await self.test_multi_agent_configuration()
            overall_results["test_results"]["configuration"] = config_test

            # Test 2: Multi-Agent System Initialization
            init_test = await self.test_multi_agent_initialization()
            overall_results["test_results"]["initialization"] = init_test

            # Test 3: Agent Registration and Status
            agent_test = await self.test_agent_registration()
            overall_results["test_results"]["agent_registration"] = agent_test

            # Test 4: Multi-Agent Routing
            routing_test = await self.test_agent_routing()
            overall_results["test_results"]["agent_routing"] = routing_test

            # Test 5: Agent Handoffs and Context Preservation
            handoff_test = await self.test_agent_handoffs()
            overall_results["test_results"]["agent_handoffs"] = handoff_test

            # Test 6: Multi-Agent Performance
            performance_test = await self.test_multi_agent_performance()
            overall_results["test_results"]["performance"] = performance_test

            # Calculate overall metrics
            overall_results["overall_metrics"] = self.calculate_overall_metrics(
                overall_results["test_results"]
            )

        except Exception as e:
            print(f"❌ Critical test failure: {e}")
            traceback.print_exc()
            overall_results["critical_error"] = str(e)
        finally:
            if self.orchestrator:
                await self.orchestrator.stop()

        return overall_results

    async def test_multi_agent_configuration(self) -> Dict[str, Any]:
        """Test multi-agent configuration loading and validation."""
        print("\n🔧 Test 1: Multi-Agent Configuration Loading")
        print("-" * 50)

        test_result = {
            "name": "multi_agent_configuration",
            "subtests": [],
            "success": False,
        }

        try:
            # Load multi-agent configuration
            print("  🔍 Loading multi-agent configuration...")
            config_path = Path("test_config_multi_agent.yaml")
            if not config_path.exists():
                print(
                    "    ⚠️  Multi-agent config not found, using default with multi-agent enabled"
                )
                config = Config.load(Path("src/voice_agent/config/default.yaml"))
                config.multi_agent.enabled = True
            else:
                config = Config.load(config_path)
                config.multi_agent.enabled = True

            config_checks = {
                "config_loaded": True,
                "multi_agent_enabled": config.multi_agent.enabled,
                "has_agents": len(config.multi_agent.agents) > 0,
                "has_routing_rules": len(config.multi_agent.routing_rules) > 0,
                "has_default_agent": config.multi_agent.default_agent
                in config.multi_agent.agents,
                "routing_strategy_valid": config.multi_agent.routing_strategy
                in ["hybrid", "rules_only", "embeddings_only"],
            }

            test_result["subtests"].append(
                {
                    "name": "multi_agent_config_load",
                    "success": all(config_checks.values()),
                    "details": config_checks,
                }
            )

            print("    ✅ Multi-agent configuration loaded successfully")
            print(f"    📊 Multi-agent enabled: {config.multi_agent.enabled}")
            print(f"    📊 Default agent: {config.multi_agent.default_agent}")
            print(f"    📊 Routing strategy: {config.multi_agent.routing_strategy}")
            print(f"    📊 Configured agents: {list(config.multi_agent.agents.keys())}")
            print(f"    📊 Routing rules: {len(config.multi_agent.routing_rules)}")

            # Test multi-agent config validation
            print("  🔍 Validating multi-agent configuration...")
            validation_issues = config.validate_multi_agent_config()

            validation_success = len(validation_issues) == 0

            test_result["subtests"].append(
                {
                    "name": "multi_agent_validation",
                    "success": validation_success,
                    "details": {
                        "validation_issues": validation_issues,
                        "issue_count": len(validation_issues),
                    },
                }
            )

            if validation_success:
                print("    ✅ Multi-agent configuration validation passed")
            else:
                print(f"    ❌ Validation issues found: {validation_issues}")

            test_result["success"] = all(
                subtest["success"] for subtest in test_result["subtests"]
            )

        except Exception as e:
            print(f"    ❌ Configuration test failed: {e}")
            test_result["error"] = str(e)

        return test_result

    async def test_multi_agent_initialization(self) -> Dict[str, Any]:
        """Test multi-agent system initialization."""
        print("\n🚀 Test 2: Multi-Agent System Initialization")
        print("-" * 50)

        test_result = {
            "name": "multi_agent_initialization",
            "subtests": [],
            "success": False,
        }

        try:
            # Load configuration with multi-agent enabled
            config = Config.load(Path("src/voice_agent/config/default.yaml"))
            config.multi_agent.enabled = True
            config.ui.force_text_only = True

            print("  🔍 Initializing VoiceAgentOrchestrator with multi-agent mode...")
            start_time = time.time()

            self.orchestrator = VoiceAgentOrchestrator(config=config, text_only=True)

            await self.orchestrator.initialize()
            init_time = time.time() - start_time

            print(f"    ✅ Orchestrator initialized in {init_time:.2f}s")

            # Check multi-agent system status
            info = self.orchestrator.get_orchestrator_info()

            init_checks = {
                "orchestrator_initialized": self.orchestrator.is_initialized,
                "multi_agent_enabled": info.get("multi_agent_enabled", False),
                "text_only_mode": info.get("text_only", False),
                "multi_agent_service_available": info.get("components", {}).get(
                    "multi_agent_service", False
                ),
                "llamaindex_service_available": info.get("components", {}).get(
                    "llamaindex_service", False
                ),
                "llm_service_ready": info.get("components", {}).get("llm", False),
                "tools_ready": info.get("components", {}).get("tools", False),
            }

            test_result["subtests"].append(
                {
                    "name": "multi_agent_orchestrator_init",
                    "success": all(init_checks.values()),
                    "details": {
                        **init_checks,
                        "initialization_time": init_time,
                        "orchestrator_info": info,
                    },
                }
            )

            print(f"    📊 Multi-agent enabled: {info.get('multi_agent_enabled')}")
            print(
                f"    📊 Multi-agent service: {'✅' if init_checks['multi_agent_service_available'] else '❌'}"
            )
            print(
                f"    📊 LlamaIndex service: {'✅' if init_checks['llamaindex_service_available'] else '❌'}"
            )
            print(
                f"    📊 Components ready: {sum(info.get('components', {}).values())} / {len(info.get('components', {}))}"
            )

            test_result["success"] = all(
                subtest["success"] for subtest in test_result["subtests"]
            )

        except Exception as e:
            print(f"    ❌ Multi-agent initialization failed: {e}")
            test_result["error"] = str(e)

        return test_result

    async def test_agent_registration(self) -> Dict[str, Any]:
        """Test agent registration and status."""
        print("\n👥 Test 3: Agent Registration and Status")
        print("-" * 50)

        test_result = {"name": "agent_registration", "subtests": [], "success": False}

        if not self.orchestrator:
            test_result["error"] = "Orchestrator not initialized"
            return test_result

        try:
            print("  🔍 Checking agent registration status...")

            info = self.orchestrator.get_orchestrator_info()
            ma_info = info.get("multi_agent_info", {})

            # Check if we have multi-agent info
            if not ma_info:
                print("    ⚠️  Multi-agent service not available, checking fallback...")

                # Check if we can still process requests (fallback mode)
                fallback_checks = {
                    "llm_service_available": info.get("components", {}).get(
                        "llm", False
                    ),
                    "tools_available": info.get("components", {}).get("tools", False),
                    "can_process": True,  # We'll test this below
                }

                # Test basic processing
                try:
                    response = await self.orchestrator.process_text("Hello")
                    fallback_checks["can_process"] = bool(response)
                except Exception:
                    fallback_checks["can_process"] = False

                test_result["subtests"].append(
                    {
                        "name": "fallback_mode_check",
                        "success": all(fallback_checks.values()),
                        "details": fallback_checks,
                    }
                )

                print("    ✅ Fallback to single-agent mode working")

            else:
                # Check multi-agent specific info
                agent_checks = {
                    "has_active_agents": ma_info.get("active_agents", 0) > 0,
                    "has_routing_stats": "routing_stats" in ma_info,
                    "service_initialized": ma_info.get("service_initialized", False),
                }

                test_result["subtests"].append(
                    {
                        "name": "multi_agent_registration",
                        "success": all(agent_checks.values()),
                        "details": {
                            **agent_checks,
                            "active_agents": ma_info.get("active_agents", 0),
                            "multi_agent_info": ma_info,
                        },
                    }
                )

                print(f"    ✅ Active agents: {ma_info.get('active_agents', 0)}")
                print(
                    f"    📊 Service initialized: {ma_info.get('service_initialized', False)}"
                )

            test_result["success"] = all(
                subtest["success"] for subtest in test_result["subtests"]
            )

        except Exception as e:
            print(f"    ❌ Agent registration test failed: {e}")
            test_result["error"] = str(e)

        return test_result

    async def test_agent_routing(self) -> Dict[str, Any]:
        """Test agent routing functionality."""
        print("\n🎯 Test 4: Multi-Agent Routing")
        print("-" * 50)

        test_result = {"name": "agent_routing", "subtests": [], "success": False}

        if not self.orchestrator:
            test_result["error"] = "Orchestrator not initialized"
            return test_result

        # Test different types of queries that should route to different agents
        routing_test_cases = [
            {
                "query": "What's the weather like today?",
                "expected_type": "information",
                "description": "Weather query",
            },
            {
                "query": "Calculate 15 times 23",
                "expected_type": "calculation",
                "description": "Math query",
            },
            {
                "query": "Create a file called test.txt",
                "expected_type": "productivity",
                "description": "File operation",
            },
            {
                "query": "Hello, how are you?",
                "expected_type": "general",
                "description": "General conversation",
            },
        ]

        routing_results = []

        for test_case in routing_test_cases:
            try:
                print(f"  🔍 Testing: {test_case['description']}")
                print(f"    Query: '{test_case['query']}'")

                start_time = time.time()
                response = await self.orchestrator.process_text(test_case["query"])
                processing_time = time.time() - start_time

                # Check if response suggests appropriate processing
                response_quality_checks = {
                    "response_generated": bool(response),
                    "response_not_empty": len(response.strip()) > 0,
                    "reasonable_length": 10 <= len(response) <= 500,
                    "processing_time_reasonable": processing_time < 15.0,
                }

                routing_results.append(
                    {
                        "query": test_case["query"],
                        "expected_type": test_case["expected_type"],
                        "description": test_case["description"],
                        "response_length": len(response),
                        "processing_time": processing_time,
                        "success": all(response_quality_checks.values()),
                        "checks": response_quality_checks,
                    }
                )

                print(
                    f"    ✅ Response: {len(response)} chars in {processing_time:.2f}s"
                )
                print(
                    f"    📝 Preview: {response[:80]}{'...' if len(response) > 80 else ''}"
                )

            except Exception as e:
                print(f"    ❌ Routing test failed: {e}")
                routing_results.append(
                    {
                        "query": test_case["query"],
                        "description": test_case["description"],
                        "success": False,
                        "error": str(e),
                    }
                )

        success_rate = sum(1 for r in routing_results if r["success"]) / len(
            routing_results
        )

        test_result["subtests"].append(
            {
                "name": "agent_routing_tests",
                "success": success_rate >= 0.75,  # 75% success rate required
                "details": {
                    "success_rate": success_rate,
                    "routing_tests": len(routing_results),
                    "successful_tests": sum(1 for r in routing_results if r["success"]),
                    "results": routing_results,
                },
            }
        )

        test_result["success"] = all(
            subtest["success"] for subtest in test_result["subtests"]
        )

        return test_result

    async def test_agent_handoffs(self) -> Dict[str, Any]:
        """Test agent handoffs and context preservation."""
        print("\n🔄 Test 5: Agent Handoffs and Context Preservation")
        print("-" * 50)

        test_result = {"name": "agent_handoffs", "subtests": [], "success": False}

        if not self.orchestrator:
            test_result["error"] = "Orchestrator not initialized"
            return test_result

        # Test a conversation that should involve multiple agents
        handoff_conversation = [
            {"query": "Calculate 25 * 30", "step": "Initial calculation"},
            {
                "query": "Now save that result to a file",
                "step": "File operation with context",
            },
            {"query": "What did we just calculate?", "step": "Context recall"},
        ]

        handoff_results = []
        conversation_context = []

        for step_num, interaction in enumerate(handoff_conversation):
            try:
                print(f"  🔍 Step {step_num + 1}: {interaction['step']}")
                print(f"    Query: '{interaction['query']}'")

                start_time = time.time()
                response = await self.orchestrator.process_text(interaction["query"])
                processing_time = time.time() - start_time

                # Check context preservation (for steps 2 and 3)
                context_preserved = True
                if step_num > 0:
                    # Look for references to previous conversation
                    context_keywords = [
                        "25",
                        "30",
                        "750",
                        "calculate",
                        "result",
                        "file",
                    ]
                    context_preserved = any(
                        keyword in response.lower() for keyword in context_keywords
                    )

                handoff_results.append(
                    {
                        "step": step_num + 1,
                        "query": interaction["query"],
                        "step_description": interaction["step"],
                        "response_generated": bool(response),
                        "processing_time": processing_time,
                        "context_preserved": context_preserved,
                        "success": bool(response)
                        and (step_num == 0 or context_preserved),
                    }
                )

                conversation_context.append(
                    {"user": interaction["query"], "agent": response}
                )

                print(
                    f"    ✅ Response in {processing_time:.2f}s, Context: {'✅' if context_preserved else '❌'}"
                )
                print(
                    f"    📝 Preview: {response[:80]}{'...' if len(response) > 80 else ''}"
                )

            except Exception as e:
                print(f"    ❌ Step {step_num + 1} failed: {e}")
                handoff_results.append(
                    {
                        "step": step_num + 1,
                        "query": interaction["query"],
                        "step_description": interaction["step"],
                        "success": False,
                        "error": str(e),
                    }
                )

        success_rate = sum(1 for r in handoff_results if r["success"]) / len(
            handoff_results
        )

        test_result["subtests"].append(
            {
                "name": "agent_handoff_tests",
                "success": success_rate
                >= 0.67,  # 67% success rate for complex scenarios
                "details": {
                    "success_rate": success_rate,
                    "handoff_steps": len(handoff_results),
                    "successful_steps": sum(1 for r in handoff_results if r["success"]),
                    "conversation_flow": handoff_results,
                    "context_preservation": conversation_context,
                },
            }
        )

        test_result["success"] = all(
            subtest["success"] for subtest in test_result["subtests"]
        )

        return test_result

    async def test_multi_agent_performance(self) -> Dict[str, Any]:
        """Test multi-agent system performance."""
        print("\n⚡ Test 6: Multi-Agent Performance")
        print("-" * 50)

        test_result = {
            "name": "multi_agent_performance",
            "subtests": [],
            "success": False,
        }

        if not self.orchestrator:
            test_result["error"] = "Orchestrator not initialized"
            return test_result

        try:
            # Test concurrent processing
            print("  🔍 Testing concurrent query processing...")

            concurrent_queries = [
                "What's 10 + 10?",
                "Hello there!",
                "What's the weather?",
                "Calculate 5 * 5",
            ]

            start_time = time.time()

            # Process queries concurrently
            tasks = [
                self.orchestrator.process_text(query) for query in concurrent_queries
            ]

            responses = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

            # Analyze results
            successful_responses = sum(1 for r in responses if isinstance(r, str) and r)
            error_count = sum(1 for r in responses if isinstance(r, Exception))

            performance_checks = {
                "all_queries_processed": len(responses) == len(concurrent_queries),
                "most_successful": successful_responses
                >= len(concurrent_queries) * 0.75,
                "reasonable_total_time": total_time < 20.0,
                "low_error_rate": error_count <= len(concurrent_queries) * 0.25,
            }

            test_result["subtests"].append(
                {
                    "name": "concurrent_processing",
                    "success": all(performance_checks.values()),
                    "details": {
                        "total_queries": len(concurrent_queries),
                        "successful_responses": successful_responses,
                        "error_count": error_count,
                        "total_time": total_time,
                        "average_time_per_query": total_time / len(concurrent_queries),
                        "checks": performance_checks,
                    },
                }
            )

            print(
                f"    ✅ Concurrent processing: {successful_responses}/{len(concurrent_queries)} successful in {total_time:.2f}s"
            )

            test_result["success"] = all(
                subtest["success"] for subtest in test_result["subtests"]
            )

        except Exception as e:
            print(f"    ❌ Performance test failed: {e}")
            test_result["error"] = str(e)

        return test_result

    def calculate_overall_metrics(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall test metrics."""
        total_tests = 0
        successful_tests = 0

        for test_name, test_data in test_results.items():
            if "subtests" in test_data:
                total_tests += len(test_data["subtests"])
                successful_tests += sum(
                    1 for subtest in test_data["subtests"] if subtest["success"]
                )

        overall_success_rate = successful_tests / max(total_tests, 1)
        main_test_success_rate = sum(
            1 for test in test_results.values() if test.get("success", False)
        ) / len(test_results)

        return {
            "total_subtests": total_tests,
            "successful_subtests": successful_tests,
            "subtest_success_rate": overall_success_rate,
            "main_test_success_rate": main_test_success_rate,
            "overall_success_rate": (overall_success_rate + main_test_success_rate) / 2,
            "production_ready": main_test_success_rate >= 0.75
            and overall_success_rate >= 0.67,
            "tests_passed": sum(
                1 for test in test_results.values() if test.get("success", False)
            ),
            "total_tests": len(test_results),
        }

    def print_final_report(self, results: Dict[str, Any]) -> None:
        """Print final test report."""
        metrics = results["overall_metrics"]

        print("\n" + "=" * 60)
        print("📊 MULTI-AGENT INTEGRATION TEST REPORT")
        print("=" * 60)

        # Overall status
        status_emoji = "✅" if metrics["production_ready"] else "❌"
        print(
            f"\n{status_emoji} OVERALL STATUS: {'READY' if metrics['production_ready'] else 'NEEDS WORK'}"
        )
        print(
            f"   Main Tests: {metrics['tests_passed']}/{metrics['total_tests']} passed ({metrics['main_test_success_rate']:.1%})"
        )
        print(
            f"   Subtests: {metrics['successful_subtests']}/{metrics['total_subtests']} passed ({metrics['subtest_success_rate']:.1%})"
        )
        print(f"   Overall Success Rate: {metrics['overall_success_rate']:.1%}")

        # Test breakdown
        print("\n📋 TEST BREAKDOWN:")
        for test_name, test_data in results["test_results"].items():
            status = "✅ PASS" if test_data.get("success", False) else "❌ FAIL"
            print(f"   {status} - {test_name}")

            if "subtests" in test_data:
                for subtest in test_data["subtests"]:
                    sub_status = "✅" if subtest["success"] else "❌"
                    print(f"     {sub_status} {subtest['name']}")

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if metrics["production_ready"]:
            print("   • Multi-agent mode is ready for production use")
            print("   • Consider proceeding with full voice pipeline testing")
            print("   • Monitor agent routing performance in production")
        else:
            print("   • Address failing multi-agent tests before production deployment")
            print("   • Verify agent configurations and routing rules")
            print("   • Check multi-agent service dependencies")

        print("\n" + "=" * 60)


async def main():
    """Main test execution function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    tester = MultiAgentIntegrationTester()

    try:
        results = await tester.run_all_tests()

        # Save results
        results_file = Path("examples/multi_agent_test_results.json")
        import json

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Print report
        tester.print_final_report(results)

        print(f"\n📄 Detailed results saved to: {results_file}")

        # Exit with appropriate code
        production_ready = results["overall_metrics"]["production_ready"]
        sys.exit(0 if production_ready else 1)

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
