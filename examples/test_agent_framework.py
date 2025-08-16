#!/usr/bin/env python3
"""
Comprehensive Agent Testing Framework

This framework provides utilities and base classes for testing individual agents
and their tools, including mock implementations, performance benchmarking,
and error simulation capabilities.
"""

import asyncio
import logging
import sys
import time
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import tempfile
import shutil

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from voice_agent.core.config import Config
from voice_agent.core.tool_executor import ToolExecutor
from voice_agent.core.multi_agent_service import MultiAgentService
from voice_agent.core.multi_agent.agent_base import AgentMessage, MessageType


class TestSeverity(Enum):
    """Test severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TestCategory(Enum):
    """Test categories for organization."""

    INITIALIZATION = "initialization"
    FUNCTIONALITY = "functionality"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    ERROR_HANDLING = "error_handling"
    EDGE_CASES = "edge_cases"
    SECURITY = "security"


@dataclass
class TestResult:
    """Test result data structure."""

    test_name: str
    category: TestCategory
    severity: TestSeverity
    passed: bool
    duration_ms: float
    details: str
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    """Performance measurement data."""

    response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    error_count: int
    total_requests: int


class MockToolExecutor:
    """Mock tool executor for testing without real external dependencies."""

    def __init__(self):
        self.tools = {}
        self.execution_history = []
        self.should_fail = False
        self.failure_message = "Mock tool execution failed"
        self.response_delay = 0.0

    async def initialize(self):
        """Mock initialization."""
        await asyncio.sleep(0.001)  # Simulate initialization delay

    async def cleanup(self):
        """Mock cleanup."""
        pass

    def register_tool(self, tool):
        """Register a mock tool."""
        self.tools[tool.name] = tool

    async def execute_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a mock tool."""
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)

        self.execution_history.append(
            {"tool_name": tool_name, "parameters": parameters, "timestamp": time.time()}
        )

        if self.should_fail:
            return {"success": False, "error": self.failure_message, "result": None}

        # Return mock successful responses based on tool type
        if tool_name == "calculator":
            return {
                "success": True,
                "result": "42",  # Mock calculation result
                "operation": parameters.get("expression", "2+2"),
            }
        elif tool_name == "weather":
            return {
                "success": True,
                "result": {
                    "location": parameters.get("location", "New York"),
                    "temperature": "22°C",
                    "condition": "Sunny",
                    "humidity": "60%",
                },
            }
        elif tool_name == "file_ops":
            return {
                "success": True,
                "result": "File operation completed successfully",
                "operation": parameters.get("operation", "read"),
            }
        elif tool_name == "web_search":
            return {
                "success": True,
                "result": [
                    {"title": "Mock Result 1", "url": "http://example.com/1"},
                    {"title": "Mock Result 2", "url": "http://example.com/2"},
                ],
                "query": parameters.get("query", "test query"),
            }
        else:
            return {
                "success": True,
                "result": f"Mock result for {tool_name}",
                "tool_name": tool_name,
            }

    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available mock tools."""
        return [
            {"name": name, "description": f"Mock {name} tool"}
            for name in self.tools.keys()
        ]

    def set_failure_mode(self, should_fail: bool, message: str = "Mock failure"):
        """Set mock tool to fail."""
        self.should_fail = should_fail
        self.failure_message = message

    def set_response_delay(self, delay_seconds: float):
        """Set artificial response delay."""
        self.response_delay = delay_seconds


class AgentTestFramework:
    """Main testing framework for agents."""

    def __init__(self, devenv_mode: bool = True):
        self.devenv_mode = devenv_mode
        self.test_results: List[TestResult] = []
        self.temp_dir = None
        self.mock_executor = MockToolExecutor()
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Set up test logging."""
        logger = logging.getLogger("agent_test_framework")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def setup_test_environment(self) -> Dict[str, Any]:
        """Set up test environment with temporary directories and mock services."""
        self.temp_dir = tempfile.mkdtemp(prefix="agent_test_")
        self.logger.info(f"Created temporary test directory: {self.temp_dir}")

        # Load test configuration
        config_path = Path("src/voice_agent/config/default.yaml")
        config = Config.load(config_path)
        config.multi_agent.enabled = True

        # Initialize mock tool executor
        await self.mock_executor.initialize()

        return {
            "config": config,
            "tool_executor": self.mock_executor,
            "temp_dir": self.temp_dir,
        }

    async def cleanup_test_environment(self):
        """Clean up test environment."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            self.logger.info(f"Cleaned up temporary directory: {self.temp_dir}")

        await self.mock_executor.cleanup()

    async def run_test(
        self,
        test_func: Callable,
        test_name: str,
        category: TestCategory,
        severity: TestSeverity,
        timeout_seconds: float = 30.0,
        **kwargs,
    ) -> TestResult:
        """Run a single test with timing and error handling."""
        start_time = time.time()

        try:
            self.logger.info(f"Running test: {test_name}")

            # Run test with timeout
            result = await asyncio.wait_for(
                test_func(**kwargs), timeout=timeout_seconds
            )

            duration_ms = (time.time() - start_time) * 1000

            test_result = TestResult(
                test_name=test_name,
                category=category,
                severity=severity,
                passed=(
                    result.get("success", False)
                    if isinstance(result, dict)
                    else bool(result)
                ),
                duration_ms=duration_ms,
                details=(
                    result.get("details", str(result))
                    if isinstance(result, dict)
                    else str(result)
                ),
                metrics=result.get("metrics") if isinstance(result, dict) else None,
            )

            self.test_results.append(test_result)
            return test_result

        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Test timeout after {timeout_seconds} seconds"

            test_result = TestResult(
                test_name=test_name,
                category=category,
                severity=severity,
                passed=False,
                duration_ms=duration_ms,
                details="Test timed out",
                error=error_msg,
            )

            self.test_results.append(test_result)
            return test_result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"{type(e).__name__}: {str(e)}"

            test_result = TestResult(
                test_name=test_name,
                category=category,
                severity=severity,
                passed=False,
                duration_ms=duration_ms,
                details="Test threw exception",
                error=error_msg,
            )

            self.test_results.append(test_result)
            self.logger.error(f"Test {test_name} failed: {error_msg}")
            self.logger.debug(traceback.format_exc())

            return test_result

    async def benchmark_agent_performance(
        self, agent, test_queries: List[str], iterations: int = 10
    ) -> PerformanceMetrics:
        """Benchmark agent performance with multiple queries."""
        response_times = []
        error_count = 0
        total_requests = iterations * len(test_queries)

        start_memory = self._get_memory_usage()

        for iteration in range(iterations):
            for query in test_queries:
                start_time = time.time()

                try:
                    message = AgentMessage(
                        conversation_id=f"benchmark_{iteration}",
                        type=MessageType.USER_INPUT,
                        content=query,
                    )

                    response = await agent.process_message(message)
                    response_time = (time.time() - start_time) * 1000
                    response_times.append(response_time)

                    if not response.success:
                        error_count += 1

                except Exception:
                    error_count += 1
                    response_times.append((time.time() - start_time) * 1000)

        end_memory = self._get_memory_usage()
        avg_response_time = sum(response_times) / len(response_times)
        success_rate = (total_requests - error_count) / total_requests

        return PerformanceMetrics(
            response_time_ms=avg_response_time,
            memory_usage_mb=end_memory - start_memory,
            cpu_usage_percent=0.0,  # Would need psutil for real CPU measurement
            success_rate=success_rate,
            error_count=error_count,
            total_requests=total_requests,
        )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB (simplified)."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0  # Return 0 if psutil not available

    async def test_agent_error_handling(
        self, agent, error_scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Test agent error handling with various failure scenarios."""
        results = {}

        for scenario in error_scenarios:
            scenario_name = scenario["name"]
            scenario_setup = scenario.get("setup", lambda: None)
            scenario_input = scenario["input"]
            expected_behavior = scenario.get("expected_behavior", "graceful_failure")

            try:
                # Set up scenario
                if callable(scenario_setup):
                    scenario_setup()

                # Execute test
                message = AgentMessage(
                    conversation_id=f"error_test_{scenario_name}",
                    type=MessageType.USER_INPUT,
                    content=scenario_input,
                )

                response = await agent.process_message(message)

                # Evaluate response
                if expected_behavior == "graceful_failure":
                    success = not response.success and response.error is not None
                elif expected_behavior == "recovery":
                    success = response.success or (
                        response.error and "retry" in response.error.lower()
                    )
                else:
                    success = response.success

                results[scenario_name] = {
                    "success": success,
                    "response": response,
                    "details": f"Expected {expected_behavior}, got {'success' if response.success else 'failure'}",
                }

            except Exception as e:
                results[scenario_name] = {
                    "success": False,
                    "error": str(e),
                    "details": f"Exception during error handling test: {e}",
                }

        return results

    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        if not self.test_results:
            return {"error": "No test results available"}

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.passed)
        failed_tests = total_tests - passed_tests

        # Group by category
        category_stats = {}
        for result in self.test_results:
            category = result.category.value
            if category not in category_stats:
                category_stats[category] = {"total": 0, "passed": 0, "failed": 0}

            category_stats[category]["total"] += 1
            if result.passed:
                category_stats[category]["passed"] += 1
            else:
                category_stats[category]["failed"] += 1

        # Group by severity
        severity_stats = {}
        for result in self.test_results:
            severity = result.severity.value
            if severity not in severity_stats:
                severity_stats[severity] = {"total": 0, "passed": 0, "failed": 0}

            severity_stats[severity]["total"] += 1
            if result.passed:
                severity_stats[severity]["passed"] += 1
            else:
                severity_stats[severity]["failed"] += 1

        # Performance stats
        durations = [result.duration_ms for result in self.test_results]
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)

        # Failed tests details
        failed_tests_details = [
            {
                "name": result.test_name,
                "category": result.category.value,
                "severity": result.severity.value,
                "error": result.error,
                "details": result.details,
            }
            for result in self.test_results
            if not result.passed
        ]

        return {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            },
            "category_breakdown": category_stats,
            "severity_breakdown": severity_stats,
            "performance": {
                "average_duration_ms": avg_duration,
                "max_duration_ms": max_duration,
                "min_duration_ms": min_duration,
            },
            "failed_tests": failed_tests_details,
            "all_results": [
                {
                    "name": result.test_name,
                    "category": result.category.value,
                    "severity": result.severity.value,
                    "passed": result.passed,
                    "duration_ms": result.duration_ms,
                    "details": result.details,
                    "error": result.error,
                }
                for result in self.test_results
            ],
        }

    def save_test_report(self, output_path: str):
        """Save test report to file."""
        report = self.generate_test_report()

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Test report saved to: {output_path}")

    def print_test_summary(self):
        """Print a formatted test summary to console."""
        report = self.generate_test_report()

        print("\n" + "=" * 80)
        print("AGENT TEST FRAMEWORK - COMPREHENSIVE RESULTS")
        print("=" * 80)

        summary = report["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']} ({summary['success_rate']:.1%})")
        print(f"Failed: {summary['failed']}")

        print(f"\nPerformance:")
        perf = report["performance"]
        print(f"  Average Duration: {perf['average_duration_ms']:.2f}ms")
        print(f"  Max Duration: {perf['max_duration_ms']:.2f}ms")
        print(f"  Min Duration: {perf['min_duration_ms']:.2f}ms")

        print(f"\nBy Category:")
        for category, stats in report["category_breakdown"].items():
            success_rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
            print(
                f"  {category:15} {stats['passed']}/{stats['total']} ({success_rate:.1%})"
            )

        print(f"\nBy Severity:")
        for severity, stats in report["severity_breakdown"].items():
            success_rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
            print(
                f"  {severity:15} {stats['passed']}/{stats['total']} ({success_rate:.1%})"
            )

        if report["failed_tests"]:
            print(f"\nFailed Tests:")
            for failure in report["failed_tests"]:
                print(f"  ❌ {failure['name']} ({failure['severity']})")
                print(f"     {failure['error'] or failure['details']}")

        print("=" * 80)


# Utility functions for specific agent testing


async def create_test_multi_agent_service(
    config: Config, tool_executor: ToolExecutor
) -> MultiAgentService:
    """Create a multi-agent service for testing."""
    service = MultiAgentService(config=config, tool_executor=tool_executor)
    await service.initialize()
    return service


def create_test_queries_for_agent(agent_type: str) -> List[str]:
    """Create test queries specific to an agent type."""
    queries = {
        "information": [
            "What's the weather in New York?",
            "Search for Python tutorials",
            "Get the latest news about AI",
            "Tell me the temperature in London",
            "Find information about renewable energy",
        ],
        "utility": [
            "Calculate 25 * 43 + 17",
            "What is 15% of 200?",
            "Solve 2^8",
            "Find the square root of 144",
            "Convert 0.75 to percentage",
        ],
        "productivity": [
            "Create a calendar event for tomorrow",
            "List files in the current directory",
            "Save this text to a file",
            "Schedule a meeting for next week",
            "Read the contents of a file",
        ],
    }
    return queries.get(agent_type, ["Hello", "How are you?", "Help me with something"])


def create_error_scenarios_for_agent(agent_type: str) -> List[Dict[str, Any]]:
    """Create error scenarios specific to an agent type."""
    base_scenarios = [
        {"name": "empty_input", "input": "", "expected_behavior": "graceful_failure"},
        {
            "name": "very_long_input",
            "input": "x" * 10000,
            "expected_behavior": "graceful_failure",
        },
        {
            "name": "special_characters",
            "input": "!@#$%^&*(){}[]|\\:;\"'<>,.?/~`",
            "expected_behavior": "graceful_failure",
        },
    ]

    agent_specific = {
        "information": [
            {
                "name": "invalid_location",
                "input": "What's the weather in NONEXISTENTCITY12345?",
                "expected_behavior": "graceful_failure",
            }
        ],
        "utility": [
            {
                "name": "division_by_zero",
                "input": "Calculate 10 / 0",
                "expected_behavior": "graceful_failure",
            },
            {
                "name": "invalid_expression",
                "input": "Calculate 2 + + 3",
                "expected_behavior": "graceful_failure",
            },
        ],
        "productivity": [
            {
                "name": "invalid_file_path",
                "input": "Read file /nonexistent/path/file.txt",
                "expected_behavior": "graceful_failure",
            }
        ],
    }

    return base_scenarios + agent_specific.get(agent_type, [])


if __name__ == "__main__":
    print("Agent Testing Framework")
    print("This module provides utilities for comprehensive agent testing.")
    print("Import this module to use the AgentTestFramework class.")
