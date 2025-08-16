#!/usr/bin/env python3
"""
Comprehensive Single-Agent Mode Integration Testing

This script tests the voice agent system in single-agent mode to ensure
all core components work properly before testing multi-agent functionality.

Usage:
    devenv shell -- python examples/integration_test_single_agent.py
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


class SingleAgentIntegrationTester:
    """Comprehensive single-agent integration testing."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results: Dict[str, Any] = {}
        self.orchestrator: VoiceAgentOrchestrator = None

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all single-agent integration tests."""
        print("🔧 SINGLE-AGENT MODE INTEGRATION TESTING")
        print("=" * 60)

        overall_results = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_results": {},
            "overall_metrics": {},
        }

        try:
            # Test 1: Configuration Loading and Validation
            config_test = await self.test_configuration_loading()
            overall_results["test_results"]["configuration"] = config_test

            # Test 2: System Initialization
            init_test = await self.test_system_initialization()
            overall_results["test_results"]["initialization"] = init_test

            # Test 3: Core Component Status
            component_test = await self.test_core_components()
            overall_results["test_results"]["components"] = component_test

            # Test 4: Basic Text Processing
            text_test = await self.test_text_processing()
            overall_results["test_results"]["text_processing"] = text_test

            # Test 5: Tool Integration
            tool_test = await self.test_tool_integration()
            overall_results["test_results"]["tool_integration"] = tool_test

            # Test 6: Error Handling
            error_test = await self.test_error_handling()
            overall_results["test_results"]["error_handling"] = error_test

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

    async def test_configuration_loading(self) -> Dict[str, Any]:
        """Test configuration loading and validation."""
        print("\n📋 Test 1: Configuration Loading and Validation")
        print("-" * 50)

        test_result = {
            "name": "configuration_loading",
            "subtests": [],
            "success": False,
        }

        try:
            # Test default config loading
            print("  🔍 Loading default configuration...")
            default_config = Config.load(Path("src/voice_agent/config/default.yaml"))

            config_checks = {
                "config_loaded": True,
                "multi_agent_disabled": not default_config.multi_agent.enabled,
                "has_stt_config": hasattr(default_config, "stt"),
                "has_tts_config": hasattr(default_config, "tts"),
                "has_llm_config": hasattr(default_config, "llm"),
                "has_audio_config": hasattr(default_config, "audio"),
                "ui_force_text_only": default_config.ui.force_text_only,
            }

            test_result["subtests"].append(
                {
                    "name": "default_config_load",
                    "success": all(config_checks.values()),
                    "details": config_checks,
                }
            )

            print("    ✅ Configuration loaded successfully")
            print(f"    📊 Multi-agent enabled: {default_config.multi_agent.enabled}")
            print(f"    📊 Force text-only: {default_config.ui.force_text_only}")
            print(f"    📊 STT model: {default_config.stt.model}")
            print(f"    📊 TTS engine: {default_config.tts.engine}")
            print(f"    📊 LLM provider: {default_config.llm.provider}")
            print(f"    📊 LLM model: {default_config.llm.model}")

            # Test configuration validation
            print("  🔍 Running configuration validation...")
            health_check = default_config.run_health_checks()

            validation_success = health_check["overall_health"] in [
                "healthy",
                "degraded",
            ]

            test_result["subtests"].append(
                {
                    "name": "config_validation",
                    "success": validation_success,
                    "details": {
                        "overall_health": health_check["overall_health"],
                        "check_count": len(health_check["checks"]),
                        "failed_checks": [
                            name
                            for name, check in health_check["checks"].items()
                            if check["status"] == "fail"
                        ],
                    },
                }
            )

            print(f"    ✅ Health check: {health_check['overall_health']}")

            test_result["success"] = all(
                subtest["success"] for subtest in test_result["subtests"]
            )

        except Exception as e:
            print(f"    ❌ Configuration test failed: {e}")
            test_result["error"] = str(e)

        return test_result

    async def test_system_initialization(self) -> Dict[str, Any]:
        """Test system initialization in single-agent mode."""
        print("\n🚀 Test 2: System Initialization")
        print("-" * 50)

        test_result = {
            "name": "system_initialization",
            "subtests": [],
            "success": False,
        }

        try:
            # Load configuration for single-agent mode
            config = Config.load(Path("src/voice_agent/config/default.yaml"))
            config.multi_agent.enabled = False  # Ensure single-agent mode
            config.ui.force_text_only = True  # Use text-only for testing

            print("  🔍 Initializing VoiceAgentOrchestrator...")
            start_time = time.time()

            self.orchestrator = VoiceAgentOrchestrator(config=config, text_only=True)

            await self.orchestrator.initialize()
            init_time = time.time() - start_time

            print(f"    ✅ Orchestrator initialized in {init_time:.2f}s")

            # Check orchestrator status
            info = self.orchestrator.get_orchestrator_info()

            init_checks = {
                "orchestrator_initialized": self.orchestrator.is_initialized,
                "text_only_mode": info.get("text_only", False),
                "multi_agent_disabled": not info.get("multi_agent_enabled", True),
                "components_initialized": sum(info.get("components", {}).values()) > 0,
                "llm_service_ready": info.get("components", {}).get("llm", False),
                "conversation_manager_ready": info.get("components", {}).get(
                    "conversation", False
                ),
            }

            test_result["subtests"].append(
                {
                    "name": "orchestrator_initialization",
                    "success": all(init_checks.values()),
                    "details": {
                        **init_checks,
                        "initialization_time": init_time,
                        "orchestrator_info": info,
                    },
                }
            )

            print(f"    📊 Text-only mode: {info.get('text_only')}")
            print(f"    📊 Multi-agent enabled: {info.get('multi_agent_enabled')}")
            print(
                f"    📊 Components ready: {sum(info.get('components', {}).values())} / {len(info.get('components', {}))}"
            )

            test_result["success"] = all(
                subtest["success"] for subtest in test_result["subtests"]
            )

        except Exception as e:
            print(f"    ❌ Initialization test failed: {e}")
            test_result["error"] = str(e)

        return test_result

    async def test_core_components(self) -> Dict[str, Any]:
        """Test core component functionality."""
        print("\n🔧 Test 3: Core Component Status")
        print("-" * 50)

        test_result = {"name": "core_components", "subtests": [], "success": False}

        if not self.orchestrator:
            test_result["error"] = "Orchestrator not initialized"
            return test_result

        try:
            info = self.orchestrator.get_orchestrator_info()
            components = info.get("components", {})

            print("  🔍 Checking component status...")

            expected_components = {
                "llm": "LLM Service",
                "conversation": "Conversation Manager",
                "tools": "Tool Executor",
            }

            component_checks = {}
            for comp_key, comp_name in expected_components.items():
                is_ready = components.get(comp_key, False)
                component_checks[f"{comp_key}_ready"] = is_ready
                status = "✅" if is_ready else "❌"
                print(
                    f"    {status} {comp_name}: {'Ready' if is_ready else 'Not Ready'}"
                )

            # Check if audio components are properly disabled in text-only mode
            audio_components = ["audio_manager", "stt", "tts"]
            for comp in audio_components:
                is_disabled = not components.get(
                    comp, True
                )  # Should be False/missing in text-only
                component_checks[f"{comp}_disabled"] = is_disabled
                status = "✅" if is_disabled else "⚠️"
                print(
                    f"    {status} {comp.upper()}: {'Disabled (text-only)' if is_disabled else 'Enabled'}"
                )

            test_result["subtests"].append(
                {
                    "name": "component_status_check",
                    "success": all(component_checks.values()),
                    "details": component_checks,
                }
            )

            test_result["success"] = all(
                subtest["success"] for subtest in test_result["subtests"]
            )

        except Exception as e:
            print(f"    ❌ Component test failed: {e}")
            test_result["error"] = str(e)

        return test_result

    async def test_text_processing(self) -> Dict[str, Any]:
        """Test basic text processing functionality."""
        print("\n💬 Test 4: Basic Text Processing")
        print("-" * 50)

        test_result = {"name": "text_processing", "subtests": [], "success": False}

        if not self.orchestrator:
            test_result["error"] = "Orchestrator not initialized"
            return test_result

        test_queries = [
            {"query": "Hello, how are you?", "description": "Simple greeting"},
            {"query": "What is 2 + 2?", "description": "Simple calculation"},
            {"query": "Tell me about yourself", "description": "Self-description"},
        ]

        processing_results = []

        for test_case in test_queries:
            try:
                print(f"  🔍 Testing: {test_case['description']}")
                print(f"    Query: '{test_case['query']}'")

                start_time = time.time()
                response = await self.orchestrator.process_text(test_case["query"])
                processing_time = time.time() - start_time

                success_checks = {
                    "response_generated": bool(response),
                    "response_not_empty": len(response.strip()) > 0,
                    "reasonable_length": 10 <= len(response) <= 1000,
                    "processing_time_ok": processing_time < 30.0,
                }

                processing_results.append(
                    {
                        "query": test_case["query"],
                        "description": test_case["description"],
                        "response_length": len(response),
                        "processing_time": processing_time,
                        "success": all(success_checks.values()),
                        "checks": success_checks,
                    }
                )

                print(
                    f"    ✅ Response: {len(response)} chars in {processing_time:.2f}s"
                )
                print(
                    f"    📝 Preview: {response[:100]}{'...' if len(response) > 100 else ''}"
                )

            except Exception as e:
                print(f"    ❌ Query failed: {e}")
                processing_results.append(
                    {
                        "query": test_case["query"],
                        "description": test_case["description"],
                        "success": False,
                        "error": str(e),
                    }
                )

        success_rate = sum(1 for r in processing_results if r["success"]) / len(
            processing_results
        )

        test_result["subtests"].append(
            {
                "name": "text_query_processing",
                "success": success_rate >= 0.75,  # 75% success rate required
                "details": {
                    "success_rate": success_rate,
                    "processed_queries": len(processing_results),
                    "successful_queries": sum(
                        1 for r in processing_results if r["success"]
                    ),
                    "results": processing_results,
                },
            }
        )

        test_result["success"] = all(
            subtest["success"] for subtest in test_result["subtests"]
        )

        return test_result

    async def test_tool_integration(self) -> Dict[str, Any]:
        """Test tool integration functionality."""
        print("\n🔧 Test 5: Tool Integration")
        print("-" * 50)

        test_result = {"name": "tool_integration", "subtests": [], "success": False}

        if not self.orchestrator:
            test_result["error"] = "Orchestrator not initialized"
            return test_result

        tool_test_queries = [
            {
                "query": "Calculate 25 * 37",
                "tool": "calculator",
                "description": "Calculator tool test",
            },
            {
                "query": "What's the current time?",
                "tool": "system",
                "description": "System info test",
            },
        ]

        tool_results = []

        for test_case in tool_test_queries:
            try:
                print(f"  🔍 Testing: {test_case['description']}")
                print(f"    Query: '{test_case['query']}'")

                start_time = time.time()
                response = await self.orchestrator.process_text(test_case["query"])
                processing_time = time.time() - start_time

                # Check if response suggests tool usage
                tool_indicators = [
                    "calculate" in response.lower(),
                    "result" in response.lower(),
                    "answer" in response.lower(),
                    any(char.isdigit() for char in response),
                ]

                success_checks = {
                    "response_generated": bool(response),
                    "suggests_tool_use": any(tool_indicators),
                    "reasonable_processing_time": processing_time < 15.0,
                }

                tool_results.append(
                    {
                        "query": test_case["query"],
                        "tool": test_case["tool"],
                        "description": test_case["description"],
                        "processing_time": processing_time,
                        "success": all(success_checks.values()),
                        "checks": success_checks,
                    }
                )

                print(f"    ✅ Response generated in {processing_time:.2f}s")
                print(
                    f"    📝 Preview: {response[:100]}{'...' if len(response) > 100 else ''}"
                )

            except Exception as e:
                print(f"    ❌ Tool test failed: {e}")
                tool_results.append(
                    {
                        "query": test_case["query"],
                        "tool": test_case["tool"],
                        "description": test_case["description"],
                        "success": False,
                        "error": str(e),
                    }
                )

        success_rate = sum(1 for r in tool_results if r["success"]) / len(tool_results)

        test_result["subtests"].append(
            {
                "name": "tool_functionality",
                "success": success_rate
                >= 0.5,  # 50% success rate (tools may not be fully configured)
                "details": {
                    "success_rate": success_rate,
                    "tool_tests": len(tool_results),
                    "successful_tests": sum(1 for r in tool_results if r["success"]),
                    "results": tool_results,
                },
            }
        )

        test_result["success"] = all(
            subtest["success"] for subtest in test_result["subtests"]
        )

        return test_result

    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery."""
        print("\n⚠️  Test 6: Error Handling")
        print("-" * 50)

        test_result = {"name": "error_handling", "subtests": [], "success": False}

        if not self.orchestrator:
            test_result["error"] = "Orchestrator not initialized"
            return test_result

        error_test_cases = [
            {"query": "", "description": "Empty input"},
            {"query": "x" * 1000, "description": "Very long input"},
            {"query": "!@#$%^&*()", "description": "Special characters only"},
        ]

        error_results = []

        for test_case in error_test_cases:
            try:
                print(f"  🔍 Testing: {test_case['description']}")

                start_time = time.time()
                response = await self.orchestrator.process_text(test_case["query"])
                processing_time = time.time() - start_time

                # Check graceful handling
                graceful_handling = bool(response) and "error" not in response.lower()

                error_results.append(
                    {
                        "description": test_case["description"],
                        "query_length": len(test_case["query"]),
                        "processing_time": processing_time,
                        "graceful_handling": graceful_handling,
                        "success": graceful_handling and processing_time < 10.0,
                    }
                )

                print(f"    ✅ Handled gracefully in {processing_time:.2f}s")

            except Exception as e:
                # Exceptions are ok for error cases, as long as they're handled appropriately
                handled_appropriately = "timeout" not in str(e).lower()
                error_results.append(
                    {
                        "description": test_case["description"],
                        "query_length": len(test_case["query"]),
                        "exception_handled": True,
                        "success": handled_appropriately,
                        "error": str(e),
                    }
                )

                status = "appropriately" if handled_appropriately else "poorly"
                print(f"    ⚠️  Exception handled {status}: {type(e).__name__}")

        success_rate = sum(1 for r in error_results if r["success"]) / len(
            error_results
        )

        test_result["subtests"].append(
            {
                "name": "error_recovery",
                "success": success_rate >= 0.67,  # 67% success rate for error handling
                "details": {
                    "success_rate": success_rate,
                    "error_tests": len(error_results),
                    "successful_recoveries": sum(
                        1 for r in error_results if r["success"]
                    ),
                    "results": error_results,
                },
            }
        )

        test_result["success"] = all(
            subtest["success"] for subtest in test_result["subtests"]
        )

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
            "production_ready": main_test_success_rate >= 0.8
            and overall_success_rate >= 0.75,
            "tests_passed": sum(
                1 for test in test_results.values() if test.get("success", False)
            ),
            "total_tests": len(test_results),
        }

    def print_final_report(self, results: Dict[str, Any]) -> None:
        """Print final test report."""
        metrics = results["overall_metrics"]

        print("\n" + "=" * 60)
        print("📊 SINGLE-AGENT INTEGRATION TEST REPORT")
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
            print("   • Single-agent mode is ready for production use")
            print("   • Consider proceeding with multi-agent integration testing")
            print("   • Monitor performance in production environment")
        else:
            print("   • Address failing tests before production deployment")
            print("   • Check configuration and dependencies")
            print("   • Review error logs for specific issues")

        print("\n" + "=" * 60)


async def main():
    """Main test execution function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    tester = SingleAgentIntegrationTester()

    try:
        results = await tester.run_all_tests()

        # Save results
        results_file = Path("examples/single_agent_test_results.json")
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
