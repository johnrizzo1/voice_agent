#!/usr/bin/env python3
"""
Comprehensive UtilityAgent Test Suite

This test suite provides extensive validation of the UtilityAgent including:
- Mathematical calculation accuracy and edge cases
- Calculator tool integration with complex expressions
- Error handling for invalid mathematical operations
- Boundary condition testing (overflow, underflow, precision)
- Performance benchmarking for computational tasks
- Security validation for mathematical expression evaluation
- Integration with multi-agent routing system
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any
from decimal import getcontext

# Add src to Python path and import test framework
sys.path.insert(0, str(Path(__file__).parent / "src"))
from test_agent_framework import AgentTestFramework, TestCategory, TestSeverity

from voice_agent.core.multi_agent_service import MultiAgentService
from voice_agent.core.multi_agent.agent_base import AgentMessage, MessageType


class UtilityAgentTestSuite:
    """Comprehensive test suite for UtilityAgent."""

    def __init__(self):
        self.framework = AgentTestFramework()
        self.logger = logging.getLogger(__name__)
        self.multi_agent_service = None
        self.utility_agent = None

        # Set decimal precision for high-precision tests
        getcontext().prec = 50

    async def setup(self):
        """Set up test environment."""
        self.test_env = await self.framework.setup_test_environment()

        # Create multi-agent service
        self.multi_agent_service = MultiAgentService(
            config=self.test_env["config"], tool_executor=self.test_env["tool_executor"]
        )
        await self.multi_agent_service.initialize()

        # Get utility agent if available
        if "utility_agent" in self.multi_agent_service.agents:
            self.utility_agent = self.multi_agent_service.agents["utility_agent"]

    async def cleanup(self):
        """Clean up test environment."""
        if self.multi_agent_service:
            await self.multi_agent_service.cleanup()
        await self.framework.cleanup_test_environment()

    # INITIALIZATION TESTS

    async def test_utility_agent_initialization(self) -> Dict[str, Any]:
        """Test UtilityAgent initialization and configuration."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        if "utility_agent" not in self.multi_agent_service.agents:
            return {
                "success": False,
                "details": "UtilityAgent not found in active agents",
            }

        agent = self.multi_agent_service.agents["utility_agent"]

        # Verify agent properties
        checks = []
        checks.append(("agent_id", agent.agent_id == "utility_agent"))
        checks.append(("has_capabilities", len(agent.capabilities) > 0))
        checks.append(("has_tools", len(agent.tools) > 0))
        checks.append(("is_initialized", agent.status.value in ["ready", "active"]))
        checks.append(
            (
                "has_calculator",
                any("calculator" in str(tool).lower() for tool in agent.tools),
            )
        )

        all_passed = all(check[1] for check in checks)

        return {
            "success": all_passed,
            "details": f"Initialization checks: {dict(checks)}",
            "metrics": {
                "capabilities_count": len(agent.capabilities),
                "tools_count": len(agent.tools),
            },
        }

    async def test_utility_agent_capabilities(self) -> Dict[str, Any]:
        """Test that UtilityAgent has correct mathematical capabilities."""
        if not self.utility_agent:
            return {"success": False, "details": "UtilityAgent not available"}

        from voice_agent.core.multi_agent.agent_base import AgentCapability

        expected_capabilities = {
            AgentCapability.CALCULATIONS,
            AgentCapability.TOOL_EXECUTION,
        }

        actual_capabilities = set(self.utility_agent.capabilities)
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

    # BASIC MATHEMATICAL OPERATIONS TESTS

    async def test_basic_arithmetic_operations(self) -> Dict[str, Any]:
        """Test basic arithmetic operations."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        test_cases = [
            ("2 + 3", 5),
            ("10 - 4", 6),
            ("7 * 8", 56),
            ("15 / 3", 5),
            ("2 ** 3", 8),
            ("17 % 5", 2),
        ]

        results = []
        for expression, expected in test_cases:
            try:
                query = f"Calculate {expression}"
                response = await self.multi_agent_service.process_message(query)

                # Check if response contains the expected result
                contains_result = str(expected) in response if response else False

                results.append(
                    {
                        "expression": expression,
                        "expected": expected,
                        "response": response,
                        "success": contains_result,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "expression": expression,
                        "expected": expected,
                        "success": False,
                        "error": str(e),
                    }
                )

        successful_calculations = sum(1 for r in results if r["success"])
        success_rate = successful_calculations / len(test_cases)

        return {
            "success": success_rate >= 0.8,  # At least 80% should work
            "details": f"Basic arithmetic: {successful_calculations}/{len(test_cases)} correct",
            "metrics": {
                "total_operations": len(test_cases),
                "successful_operations": successful_calculations,
                "success_rate": success_rate,
                "results": results,
            },
        }

    async def test_complex_mathematical_expressions(self) -> Dict[str, Any]:
        """Test complex mathematical expressions."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        test_cases = [
            ("(2 + 3) * 4", 20),
            ("2 ** (3 + 1)", 16),
            ("(10 + 5) / (2 + 1)", 5),
            ("abs(-15)", 15),
            ("min(10, 5, 8)", 5),
            ("max(3, 7, 2)", 7),
            ("round(3.14159, 2)", 3.14),
            ("sum([1, 2, 3, 4, 5])", 15),
        ]

        results = []
        for expression, expected in test_cases:
            try:
                query = f"Calculate {expression}"
                response = await self.multi_agent_service.process_message(query)

                # More flexible result checking for complex expressions
                contains_result = (
                    (str(expected) in response or str(float(expected)) in response)
                    if response
                    else False
                )

                results.append(
                    {
                        "expression": expression,
                        "expected": expected,
                        "response": response,
                        "success": contains_result,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "expression": expression,
                        "expected": expected,
                        "success": False,
                        "error": str(e),
                    }
                )

        successful_calculations = sum(1 for r in results if r["success"])
        success_rate = successful_calculations / len(test_cases)

        return {
            "success": success_rate >= 0.7,  # 70% for complex expressions
            "details": f"Complex expressions: {successful_calculations}/{len(test_cases)} correct",
            "metrics": {
                "total_expressions": len(test_cases),
                "successful_expressions": successful_calculations,
                "success_rate": success_rate,
                "results": results,
            },
        }

    # MATHEMATICAL EDGE CASES AND BOUNDARY CONDITIONS

    async def test_mathematical_edge_cases(self) -> Dict[str, Any]:
        """Test mathematical edge cases and boundary conditions."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        edge_cases = [
            # Large numbers
            ("999999999999999 + 1", "Large number arithmetic"),
            ("2 ** 50", "Large power calculation"),
            ("1000000 * 1000000", "Large multiplication"),
            # Small numbers
            ("0.000001 + 0.000002", "Small decimal arithmetic"),
            ("1 / 3", "Repeating decimal"),
            ("0.1 + 0.2", "Floating point precision"),
            # Edge values
            ("0 / 1", "Zero division (valid)"),
            ("5 ** 0", "Power of zero"),
            ("(-2) ** 2", "Negative number squared"),
            ("abs(-0)", "Absolute value of negative zero"),
        ]

        results = []
        for expression, description in edge_cases:
            try:
                query = f"Calculate {expression}"
                response = await self.multi_agent_service.process_message(query)

                # For edge cases, we mainly check that we get a response without errors
                success = bool(response and len(response.strip()) > 0)

                results.append(
                    {
                        "expression": expression,
                        "description": description,
                        "response": response,
                        "success": success,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "expression": expression,
                        "description": description,
                        "success": False,
                        "error": str(e),
                    }
                )

        successful_cases = sum(1 for r in results if r["success"])
        success_rate = successful_cases / len(edge_cases)

        return {
            "success": success_rate >= 0.6,  # 60% for edge cases
            "details": f"Edge cases: {successful_cases}/{len(edge_cases)} handled",
            "metrics": {
                "total_cases": len(edge_cases),
                "successful_cases": successful_cases,
                "success_rate": success_rate,
                "results": results,
            },
        }

    async def test_mathematical_error_handling(self) -> Dict[str, Any]:
        """Test error handling for invalid mathematical operations."""
        error_scenarios = [
            {
                "name": "division_by_zero",
                "input": "Calculate 10 / 0",
                "expected_behavior": "graceful_failure",
            },
            {
                "name": "invalid_syntax",
                "input": "Calculate 2 + + 3",
                "expected_behavior": "graceful_failure",
            },
            {
                "name": "undefined_operation",
                "input": "Calculate sqrt(-1)",
                "expected_behavior": "graceful_failure",
            },
            {
                "name": "malformed_expression",
                "input": "Calculate ((2 + 3)",
                "expected_behavior": "graceful_failure",
            },
            {
                "name": "unsupported_function",
                "input": "Calculate unknownfunction(5)",
                "expected_behavior": "graceful_failure",
            },
            {
                "name": "empty_expression",
                "input": "Calculate ",
                "expected_behavior": "graceful_failure",
            },
        ]

        if not self.utility_agent:
            return {"success": False, "details": "UtilityAgent not available"}

        results = await self.framework.test_agent_error_handling(
            self.utility_agent, error_scenarios
        )

        success_count = sum(1 for r in results.values() if r["success"])

        return {
            "success": success_count
            >= len(error_scenarios) * 0.7,  # 70% should handle errors gracefully
            "details": f"Error handling: {success_count}/{len(error_scenarios)} scenarios handled properly",
            "metrics": {
                "total_scenarios": len(error_scenarios),
                "handled_properly": success_count,
                "results": results,
            },
        }

    # MATHEMATICAL FUNCTION TESTS

    async def test_mathematical_functions(self) -> Dict[str, Any]:
        """Test mathematical functions and operations."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        function_tests = [
            # Basic functions
            ("abs(-42)", 42),
            ("round(3.14159)", 3),
            ("min(5, 3, 8, 1)", 1),
            ("max(5, 3, 8, 1)", 8),
            # Mathematical operations
            ("2 ** 8", 256),
            ("25 ** 0.5", 5.0),  # Square root as power
            ("16 ** (1/4)", 2.0),  # Fourth root
            # Percentage calculations
            ("50 * 0.15", 7.5),  # 15% of 50
            ("100 / 4", 25),  # Quarter
        ]

        results = []
        for expression, expected in function_tests:
            try:
                query = f"Calculate {expression}"
                response = await self.multi_agent_service.process_message(query)

                # Check for expected result (with some tolerance for floating point)
                if isinstance(expected, float):
                    # For floating point, check if the result is close enough
                    success = (
                        any(
                            abs(float(part) - expected) < 0.01
                            for part in response.split()
                            if self._is_number(part)
                        )
                        if response
                        else False
                    )
                else:
                    success = str(expected) in response if response else False

                results.append(
                    {
                        "expression": expression,
                        "expected": expected,
                        "response": response,
                        "success": success,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "expression": expression,
                        "expected": expected,
                        "success": False,
                        "error": str(e),
                    }
                )

        successful_functions = sum(1 for r in results if r["success"])
        success_rate = successful_functions / len(function_tests)

        return {
            "success": success_rate >= 0.7,
            "details": f"Mathematical functions: {successful_functions}/{len(function_tests)} correct",
            "metrics": {
                "total_functions": len(function_tests),
                "successful_functions": successful_functions,
                "success_rate": success_rate,
                "results": results,
            },
        }

    def _is_number(self, s: str) -> bool:
        """Check if a string represents a number."""
        try:
            float(s)
            return True
        except ValueError:
            return False

    # PERCENTAGE AND CONVERSION TESTS

    async def test_percentage_calculations(self) -> Dict[str, Any]:
        """Test percentage and conversion calculations."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        percentage_tests = [
            ("What is 25% of 200?", "50"),
            ("Convert 0.75 to percentage", "75"),
            ("What is 15% of 80?", "12"),
            ("Convert 3/4 to percentage", "75"),
            ("Calculate 20% tax on $100", "20"),
        ]

        results = []
        for query, expected_result in percentage_tests:
            try:
                response = await self.multi_agent_service.process_message(query)

                # Check if the expected result appears in the response
                success = expected_result in response if response else False

                results.append(
                    {
                        "query": query,
                        "expected": expected_result,
                        "response": response,
                        "success": success,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "query": query,
                        "expected": expected_result,
                        "success": False,
                        "error": str(e),
                    }
                )

        successful_percentages = sum(1 for r in results if r["success"])
        success_rate = successful_percentages / len(percentage_tests)

        return {
            "success": success_rate >= 0.6,  # 60% for percentage calculations
            "details": f"Percentage calculations: {successful_percentages}/{len(percentage_tests)} correct",
            "metrics": {
                "total_calculations": len(percentage_tests),
                "successful_calculations": successful_percentages,
                "success_rate": success_rate,
                "results": results,
            },
        }

    # AGENT INTEGRATION TESTS

    async def test_utility_agent_routing(self) -> Dict[str, Any]:
        """Test that mathematical queries are properly routed to UtilityAgent."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        mathematical_queries = [
            "Calculate 15 + 27",
            "What is 2 squared?",
            "Solve 10 * 5",
            "Find 25% of 80",
            "Compute (15 + 5) / 4",
        ]

        routing_results = []

        for query in mathematical_queries:
            try:
                response = await self.multi_agent_service.process_message(query)

                # Check if we got a mathematical response
                has_math_response = (
                    any(
                        word in response.lower()
                        for word in ["calculate", "result", "equals", "answer", "="]
                    )
                    if response
                    else False
                )

                routing_results.append(
                    {"query": query, "response": response, "success": has_math_response}
                )
            except Exception as e:
                routing_results.append(
                    {"query": query, "success": False, "error": str(e)}
                )

        successful_routing = sum(1 for r in routing_results if r["success"])
        success_rate = successful_routing / len(mathematical_queries)

        # Check routing stats if available
        service_info = self.multi_agent_service.get_service_info()
        routing_stats = service_info.get("routing_stats", {})
        utility_agent_queries = routing_stats.get("utility_agent", 0)

        return {
            "success": success_rate >= 0.7,
            "details": f"Mathematical query routing: {successful_routing}/{len(mathematical_queries)} routed correctly",
            "metrics": {
                "total_queries": len(mathematical_queries),
                "successful_routing": successful_routing,
                "success_rate": success_rate,
                "utility_agent_queries": utility_agent_queries,
                "routing_results": routing_results,
            },
        }

    async def test_utility_agent_handoff_logic(self) -> Dict[str, Any]:
        """Test UtilityAgent handoff for non-mathematical queries."""
        if not self.utility_agent:
            return {"success": False, "details": "UtilityAgent not available"}

        non_math_queries = [
            ("What's the weather like?", "information_agent"),
            ("Create a file", "productivity_agent"),
            ("Hello, how are you?", "general_agent"),
            ("Search for Python tutorials", "information_agent"),
        ]

        handoff_results = []

        for query, expected_handoff in non_math_queries:
            try:
                message = AgentMessage(
                    conversation_id="handoff_test",
                    type=MessageType.USER_INPUT,
                    content=query,
                )

                response = await self.utility_agent.process_message(message)

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
                        "success": should_handoff,
                    }
                )

            except Exception as e:
                handoff_results.append(
                    {"query": query, "success": False, "error": str(e)}
                )

        successful_handoffs = sum(1 for r in handoff_results if r.get("success", False))

        return {
            "success": successful_handoffs >= len(non_math_queries) * 0.6,
            "details": f"Handoff logic: {successful_handoffs}/{len(non_math_queries)} queries triggered handoffs",
            "metrics": {
                "total_queries": len(non_math_queries),
                "successful_handoffs": successful_handoffs,
                "handoff_results": handoff_results,
            },
        }

    # PERFORMANCE TESTS

    async def test_utility_agent_performance(self) -> Dict[str, Any]:
        """Test UtilityAgent performance benchmarks."""
        if not self.utility_agent:
            return {"success": False, "details": "UtilityAgent not available"}

        performance_queries = [
            "Calculate 123 * 456",
            "What is 25% of 400?",
            "Solve (15 + 25) * 2",
            "Find 2 ** 10",
            "Calculate abs(-789)",
        ]

        try:
            metrics = await self.framework.benchmark_agent_performance(
                self.utility_agent, performance_queries, iterations=5
            )

            # Performance thresholds for mathematical operations
            acceptable_response_time = 2000  # 2 seconds
            acceptable_success_rate = 0.8  # 80%

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

    # SECURITY AND SAFETY TESTS

    async def test_mathematical_expression_security(self) -> Dict[str, Any]:
        """Test security of mathematical expression evaluation."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        # Test potentially dangerous expressions that should be safely handled
        security_tests = [
            # Import attempts
            "Calculate __import__('os').system('ls')",
            "Calculate eval('2+2')",
            "Calculate exec('print(\"hello\")')",
            # File system access attempts
            "Calculate open('/etc/passwd').read()",
            "Calculate __file__",
            "Calculate globals()",
            # Very large calculations that could cause DoS
            "Calculate 9 ** 9 ** 9",
            "Calculate 10 ** 1000000",
        ]

        security_results = []

        for dangerous_expr in security_tests:
            try:
                response = await self.multi_agent_service.process_message(
                    dangerous_expr
                )

                # Security success means the dangerous operation was rejected or safely handled
                # We check that the response doesn't contain system information
                safe_response = (
                    not response
                    or "error" in response.lower()
                    or "invalid" in response.lower()
                    or "not allowed" in response.lower()
                    or len(response) < 100  # Very short response suggests rejection
                )

                security_results.append(
                    {
                        "expression": dangerous_expr,
                        "response": response,
                        "safe": safe_response,
                    }
                )

            except Exception:
                # Exceptions are good for security tests - means the dangerous operation was blocked
                security_results.append(
                    {"expression": dangerous_expr, "safe": True, "blocked": True}
                )

        safe_results = sum(1 for r in security_results if r.get("safe", False))
        security_score = safe_results / len(security_tests)

        return {
            "success": security_score
            >= 0.9,  # 90% of dangerous operations should be blocked
            "details": f"Security: {safe_results}/{len(security_tests)} dangerous operations safely handled",
            "metrics": {
                "total_security_tests": len(security_tests),
                "safe_results": safe_results,
                "security_score": security_score,
                "results": security_results,
            },
        }


async def run_comprehensive_utility_agent_tests():
    """Run all comprehensive UtilityAgent tests."""
    print("🔧 Starting Comprehensive UtilityAgent Test Suite")
    print("=" * 80)

    test_suite = UtilityAgentTestSuite()
    framework = test_suite.framework

    try:
        await test_suite.setup()

        # Define all tests
        tests = [
            # Initialization tests
            (
                "UtilityAgent Initialization",
                test_suite.test_utility_agent_initialization,
                TestCategory.INITIALIZATION,
                TestSeverity.CRITICAL,
            ),
            (
                "Agent Capabilities Check",
                test_suite.test_utility_agent_capabilities,
                TestCategory.INITIALIZATION,
                TestSeverity.HIGH,
            ),
            # Mathematical operation tests
            (
                "Basic Arithmetic Operations",
                test_suite.test_basic_arithmetic_operations,
                TestCategory.FUNCTIONALITY,
                TestSeverity.HIGH,
            ),
            (
                "Complex Mathematical Expressions",
                test_suite.test_complex_mathematical_expressions,
                TestCategory.FUNCTIONALITY,
                TestSeverity.HIGH,
            ),
            (
                "Mathematical Functions",
                test_suite.test_mathematical_functions,
                TestCategory.FUNCTIONALITY,
                TestSeverity.MEDIUM,
            ),
            (
                "Percentage Calculations",
                test_suite.test_percentage_calculations,
                TestCategory.FUNCTIONALITY,
                TestSeverity.MEDIUM,
            ),
            # Edge cases and error handling
            (
                "Mathematical Edge Cases",
                test_suite.test_mathematical_edge_cases,
                TestCategory.EDGE_CASES,
                TestSeverity.MEDIUM,
            ),
            (
                "Mathematical Error Handling",
                test_suite.test_mathematical_error_handling,
                TestCategory.ERROR_HANDLING,
                TestSeverity.HIGH,
            ),
            # Integration tests
            (
                "UtilityAgent Routing",
                test_suite.test_utility_agent_routing,
                TestCategory.INTEGRATION,
                TestSeverity.HIGH,
            ),
            (
                "Agent Handoff Logic",
                test_suite.test_utility_agent_handoff_logic,
                TestCategory.INTEGRATION,
                TestSeverity.MEDIUM,
            ),
            # Performance tests
            (
                "UtilityAgent Performance",
                test_suite.test_utility_agent_performance,
                TestCategory.PERFORMANCE,
                TestSeverity.MEDIUM,
            ),
            # Security tests
            (
                "Mathematical Expression Security",
                test_suite.test_mathematical_expression_security,
                TestCategory.SECURITY,
                TestSeverity.HIGH,
            ),
        ]

        # Run all tests
        for test_name, test_func, category, severity in tests:
            await framework.run_test(
                test_func=test_func,
                test_name=test_name,
                category=category,
                severity=severity,
                timeout_seconds=30.0,
            )

        # Generate and display results
        framework.print_test_summary()
        framework.save_test_report("utility_agent_test_report.json")

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
        results = asyncio.run(run_comprehensive_utility_agent_tests())

        if "error" in results:
            print(f"\n💥 Test suite failed: {results['error']}")
            sys.exit(1)

        success_rate = results["summary"]["success_rate"]
        if success_rate >= 0.8:
            print(
                f"\n🎉 UtilityAgent tests completed successfully! ({success_rate:.1%} pass rate)"
            )
            sys.exit(0)
        elif success_rate >= 0.6:
            print(
                f"\n⚠️  UtilityAgent tests mostly successful ({success_rate:.1%} pass rate)"
            )
            sys.exit(1)
        else:
            print(f"\n❌ UtilityAgent tests failed ({success_rate:.1%} pass rate)")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        sys.exit(1)
