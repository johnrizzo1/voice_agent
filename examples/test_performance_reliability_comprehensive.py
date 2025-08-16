#!/usr/bin/env python3
"""
Comprehensive Performance and Reliability Test Suite

This test suite provides extensive validation of system performance and reliability including:
- Agent response times and performance metrics under various loads
- Concurrent agent operations and load handling capabilities
- Memory usage monitoring and resource cleanup verification
- System reliability under stress conditions and edge cases
- Error recovery and resilience testing with failure injection
- Logging and monitoring functionality validation
- Resource leak detection and cleanup verification
- Performance regression testing and benchmarking
"""

import asyncio
import logging
import sys
import time
import gc
import psutil
from pathlib import Path
from typing import Dict, Any
import threading

# Add src to Python path and import test framework
sys.path.insert(0, str(Path(__file__).parent / "src"))
from test_agent_framework import AgentTestFramework, TestCategory, TestSeverity

from voice_agent.core.multi_agent_service import MultiAgentService


class PerformanceReliabilityTestSuite:
    """Comprehensive test suite for performance and reliability."""

    def __init__(self):
        self.framework = AgentTestFramework()
        self.logger = logging.getLogger(__name__)
        self.multi_agent_service = None
        self.initial_memory = 0
        self.initial_threads = 0

    async def setup(self):
        """Set up test environment."""
        self.test_env = await self.framework.setup_test_environment()

        # Record initial system state
        self.initial_memory = self._get_memory_usage()
        self.initial_threads = threading.active_count()

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

        # Force garbage collection
        gc.collect()

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0

    def _get_thread_count(self) -> int:
        """Get current thread count."""
        return threading.active_count()

    # PERFORMANCE BENCHMARKING TESTS

    async def test_single_agent_response_times(self) -> Dict[str, Any]:
        """Test response times for individual agents under normal load."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        # Test queries for different agent types
        agent_test_queries = {
            "information": [
                "What's the weather in London?",
                "Search for Python tutorials",
                "Get tech news updates",
            ],
            "utility": ["Calculate 25 * 43 + 17", "What is 15% of 200?", "Solve 2^8"],
            "productivity": [
                "List files in current directory",
                "Schedule a meeting for tomorrow",
                "Create a task list",
            ],
        }

        performance_results = {}

        for agent_type, queries in agent_test_queries.items():
            agent_times = []
            successful_queries = 0

            for query in queries:
                start_time = time.time()

                try:
                    response = await self.multi_agent_service.process_message(query)
                    response_time = (time.time() - start_time) * 1000

                    if response and len(response.strip()) > 0:
                        successful_queries += 1
                        agent_times.append(response_time)

                except Exception as e:
                    self.logger.error(f"Query failed during performance test: {e}")
                    agent_times.append((time.time() - start_time) * 1000)

            if agent_times:
                avg_time = sum(agent_times) / len(agent_times)
                max_time = max(agent_times)
                min_time = min(agent_times)

                performance_results[agent_type] = {
                    "average_response_time_ms": avg_time,
                    "max_response_time_ms": max_time,
                    "min_response_time_ms": min_time,
                    "successful_queries": successful_queries,
                    "total_queries": len(queries),
                    "success_rate": successful_queries / len(queries),
                }

        # Calculate overall performance metrics
        all_times = []
        total_successful = 0
        total_queries = 0

        for agent_data in performance_results.values():
            # Estimate individual times from averages (approximation)
            agent_avg = agent_data["average_response_time_ms"]
            agent_queries = agent_data["successful_queries"]
            all_times.extend([agent_avg] * agent_queries)
            total_successful += agent_data["successful_queries"]
            total_queries += agent_data["total_queries"]

        overall_avg = sum(all_times) / len(all_times) if all_times else 0
        overall_success_rate = (
            total_successful / total_queries if total_queries > 0 else 0
        )

        # Performance thresholds
        acceptable_avg_time = 5000  # 5 seconds
        acceptable_success_rate = 0.8  # 80%

        performance_good = (
            overall_avg < acceptable_avg_time
            and overall_success_rate >= acceptable_success_rate
        )

        return {
            "success": performance_good,
            "details": f"Single agent performance: {overall_avg:.2f}ms avg, {overall_success_rate:.1%} success",
            "metrics": {
                "overall_average_ms": overall_avg,
                "overall_success_rate": overall_success_rate,
                "total_queries": total_queries,
                "agent_breakdown": performance_results,
                "performance_threshold_met": performance_good,
            },
        }

    async def test_concurrent_agent_operations(self) -> Dict[str, Any]:
        """Test concurrent agent operations and load handling."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        # Concurrent test queries
        concurrent_queries = [
            "What's the weather in New York?",
            "Calculate 123 * 456",
            "List my recent files",
            "Search for machine learning tutorials",
            "What is 25% of 800?",
            "Schedule a team meeting",
            "Get the latest tech news",
            "Calculate the square root of 144",
            "Create a project checklist",
            "Find information about Python",
        ]

        # Test different concurrency levels
        concurrency_levels = [1, 3, 5, 8]
        concurrency_results = {}

        for concurrency in concurrency_levels:
            start_time = time.time()
            start_memory = self._get_memory_usage()

            # Execute queries concurrently
            tasks = []
            query_subset = concurrent_queries[:concurrency]

            for i, query in enumerate(query_subset):
                task = self.multi_agent_service.process_message(
                    query, conversation_id=f"concurrent_test_{concurrency}_{i}"
                )
                tasks.append(task)

            try:
                responses = await asyncio.gather(*tasks, return_exceptions=True)

                total_time = (time.time() - start_time) * 1000
                end_memory = self._get_memory_usage()
                memory_increase = end_memory - start_memory

                # Analyze results
                successful_responses = sum(
                    1
                    for response in responses
                    if not isinstance(response, Exception)
                    and response
                    and len(response.strip()) > 0
                )

                exceptions = sum(
                    1 for response in responses if isinstance(response, Exception)
                )

                concurrency_results[concurrency] = {
                    "total_time_ms": total_time,
                    "average_time_per_query_ms": total_time / concurrency,
                    "successful_responses": successful_responses,
                    "exceptions": exceptions,
                    "success_rate": successful_responses / concurrency,
                    "memory_increase_mb": memory_increase,
                    "queries_per_second": (
                        concurrency / (total_time / 1000) if total_time > 0 else 0
                    ),
                }

            except Exception as e:
                concurrency_results[concurrency] = {
                    "error": str(e),
                    "success_rate": 0.0,
                }

        # Evaluate concurrent performance
        best_concurrency = max(
            (
                level
                for level in concurrency_results.keys()
                if isinstance(concurrency_results[level], dict)
                and "success_rate" in concurrency_results[level]
            ),
            key=lambda x: concurrency_results[x].get("success_rate", 0),
            default=1,
        )

        best_performance = concurrency_results.get(best_concurrency, {})
        overall_success = best_performance.get("success_rate", 0) >= 0.7

        return {
            "success": overall_success,
            "details": f"Concurrent operations: best at {best_concurrency} concurrent, {best_performance.get('success_rate', 0):.1%} success",
            "metrics": {
                "concurrency_levels_tested": concurrency_levels,
                "best_concurrency_level": best_concurrency,
                "best_success_rate": best_performance.get("success_rate", 0),
                "concurrency_results": concurrency_results,
                "overall_success": overall_success,
            },
        }

    # MEMORY AND RESOURCE TESTS

    async def test_memory_usage_and_cleanup(self) -> Dict[str, Any]:
        """Test memory usage patterns and resource cleanup."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        initial_memory = self._get_memory_usage()
        memory_samples = [initial_memory]

        # Execute multiple queries to build up memory usage
        test_queries = [
            "What's the weather in various cities around the world?",
            "Calculate complex mathematical expressions with multiple operations",
            "Search for comprehensive information about artificial intelligence",
            "Create detailed project documentation and file organization",
            "Generate extensive reports and summaries for business analysis",
        ] * 5  # Repeat to increase memory pressure

        peak_memory = initial_memory
        successful_queries = 0

        for i, query in enumerate(test_queries):
            try:
                conversation_id = f"memory_test_{i}"
                response = await self.multi_agent_service.process_message(
                    query, conversation_id=conversation_id
                )

                if response and len(response.strip()) > 0:
                    successful_queries += 1

                # Sample memory usage
                current_memory = self._get_memory_usage()
                memory_samples.append(current_memory)
                peak_memory = max(peak_memory, current_memory)

                # Occasional garbage collection
                if i % 5 == 0:
                    gc.collect()

            except Exception as e:
                self.logger.error(f"Memory test query failed: {e}")

        # Force cleanup and measure final memory
        gc.collect()
        await asyncio.sleep(1)  # Allow cleanup
        final_memory = self._get_memory_usage()

        # Calculate memory metrics
        memory_increase = peak_memory - initial_memory
        memory_cleanup_efficiency = (
            (peak_memory - final_memory) / memory_increase
            if memory_increase > 0
            else 1.0
        )
        average_memory = sum(memory_samples) / len(memory_samples)

        # Memory thresholds
        acceptable_peak_increase = 100  # 100MB
        acceptable_cleanup_efficiency = 0.5  # 50% cleanup
        acceptable_success_rate = 0.8  # 80%

        success_rate = successful_queries / len(test_queries)
        memory_performance_good = (
            memory_increase < acceptable_peak_increase
            and memory_cleanup_efficiency >= acceptable_cleanup_efficiency
            and success_rate >= acceptable_success_rate
        )

        return {
            "success": memory_performance_good,
            "details": f"Memory usage: {memory_increase:.1f}MB peak increase, {memory_cleanup_efficiency:.1%} cleanup efficiency",
            "metrics": {
                "initial_memory_mb": initial_memory,
                "peak_memory_mb": peak_memory,
                "final_memory_mb": final_memory,
                "memory_increase_mb": memory_increase,
                "memory_cleanup_efficiency": memory_cleanup_efficiency,
                "average_memory_mb": average_memory,
                "successful_queries": successful_queries,
                "total_queries": len(test_queries),
                "success_rate": success_rate,
                "memory_samples": memory_samples[-10:],  # Last 10 samples
                "memory_performance_good": memory_performance_good,
            },
        }

    async def test_resource_leak_detection(self) -> Dict[str, Any]:
        """Test for resource leaks (threads, file handles, etc.)."""
        initial_threads = self._get_thread_count()
        initial_memory = self._get_memory_usage()

        # Perform multiple service initialization/cleanup cycles
        leak_test_cycles = 3
        thread_counts = [initial_threads]
        memory_readings = [initial_memory]

        for cycle in range(leak_test_cycles):
            try:
                # Create and cleanup service multiple times
                temp_service = MultiAgentService(
                    config=self.test_env["config"],
                    tool_executor=self.test_env["tool_executor"],
                )

                await temp_service.initialize()

                # Do some work
                await temp_service.process_message(
                    "Test query for resource leak detection"
                )

                await temp_service.cleanup()
                del temp_service

                # Force cleanup
                gc.collect()
                await asyncio.sleep(0.5)

                # Record resource usage
                current_threads = self._get_thread_count()
                current_memory = self._get_memory_usage()

                thread_counts.append(current_threads)
                memory_readings.append(current_memory)

            except Exception as e:
                self.logger.error(f"Resource leak test cycle {cycle} failed: {e}")

        # Analyze for leaks
        final_threads = self._get_thread_count()
        final_memory = self._get_memory_usage()

        thread_leak = final_threads - initial_threads
        memory_leak = final_memory - initial_memory

        # Leak thresholds (some increase is expected)
        acceptable_thread_leak = 2  # 2 additional threads
        acceptable_memory_leak = 20  # 20MB

        no_significant_leaks = (
            thread_leak <= acceptable_thread_leak
            and memory_leak <= acceptable_memory_leak
        )

        return {
            "success": no_significant_leaks,
            "details": f"Resource leaks: {thread_leak} threads, {memory_leak:.1f}MB memory",
            "metrics": {
                "initial_threads": initial_threads,
                "final_threads": final_threads,
                "thread_leak": thread_leak,
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_leak_mb": memory_leak,
                "test_cycles": leak_test_cycles,
                "thread_counts": thread_counts,
                "memory_readings": memory_readings,
                "no_significant_leaks": no_significant_leaks,
            },
        }

    # RELIABILITY AND RESILIENCE TESTS

    async def test_error_recovery_resilience(self) -> Dict[str, Any]:
        """Test system resilience and error recovery capabilities."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        # Error injection scenarios
        error_scenarios = [
            {
                "name": "tool_failure",
                "setup": lambda: self.framework.mock_executor.set_failure_mode(
                    True, "Tool execution failed"
                ),
                "query": "Calculate 25 * 43",
                "cleanup": lambda: self.framework.mock_executor.set_failure_mode(False),
            },
            {
                "name": "slow_response",
                "setup": lambda: self.framework.mock_executor.set_response_delay(2.0),
                "query": "What's the weather today?",
                "cleanup": lambda: self.framework.mock_executor.set_response_delay(0.0),
            },
            {
                "name": "invalid_input",
                "setup": lambda: None,
                "query": "!@#$%^&*()_+{}|:<>?[]\\;'\".,/~`",
                "cleanup": lambda: None,
            },
        ]

        recovery_results = []

        for scenario in error_scenarios:
            try:
                # Setup error condition
                if scenario["setup"]:
                    scenario["setup"]()

                # Test system response
                start_time = time.time()
                response = await self.multi_agent_service.process_message(
                    scenario["query"]
                )
                response_time = (time.time() - start_time) * 1000

                # Cleanup error condition
                if scenario["cleanup"]:
                    scenario["cleanup"]()

                # Evaluate recovery
                recovered_gracefully = bool(response)  # System provided some response
                response_reasonable_time = response_time < 10000  # Under 10 seconds

                recovery_results.append(
                    {
                        "scenario": scenario["name"],
                        "query": scenario["query"],
                        "response": response,
                        "response_time_ms": response_time,
                        "recovered_gracefully": recovered_gracefully,
                        "response_reasonable_time": response_reasonable_time,
                        "success": recovered_gracefully and response_reasonable_time,
                    }
                )

            except Exception as e:
                # Cleanup in case of exception
                if scenario["cleanup"]:
                    scenario["cleanup"]()

                recovery_results.append(
                    {
                        "scenario": scenario["name"],
                        "query": scenario["query"],
                        "success": False,
                        "error": str(e),
                    }
                )

        successful_recoveries = sum(1 for r in recovery_results if r["success"])
        recovery_rate = successful_recoveries / len(error_scenarios)

        return {
            "success": recovery_rate >= 0.7,  # 70% recovery rate
            "details": f"Error recovery: {successful_recoveries}/{len(error_scenarios)} scenarios recovered gracefully",
            "metrics": {
                "total_scenarios": len(error_scenarios),
                "successful_recoveries": successful_recoveries,
                "recovery_rate": recovery_rate,
                "recovery_results": recovery_results,
            },
        }

    async def test_system_stability_under_load(self) -> Dict[str, Any]:
        """Test system stability under sustained load."""
        if not self.multi_agent_service.multi_agent_enabled:
            return {"success": False, "details": "Multi-agent service not available"}

        # Sustained load test
        load_duration_seconds = 30  # 30 second load test
        queries_per_second = 2
        total_queries = load_duration_seconds * queries_per_second

        test_queries = [
            "What's the weather?",
            "Calculate 10 + 20",
            "List my files",
            "Search for information",
        ]

        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_threads = self._get_thread_count()

        successful_queries = 0
        failed_queries = 0
        response_times = []

        try:
            for i in range(total_queries):
                query_start = time.time()
                query = test_queries[i % len(test_queries)]

                try:
                    response = await self.multi_agent_service.process_message(
                        f"{query} (load test {i})", conversation_id=f"load_test_{i}"
                    )

                    query_time = (time.time() - query_start) * 1000
                    response_times.append(query_time)

                    if response and len(response.strip()) > 0:
                        successful_queries += 1
                    else:
                        failed_queries += 1

                except Exception as e:
                    failed_queries += 1
                    self.logger.error(f"Load test query {i} failed: {e}")

                # Maintain roughly consistent query rate
                elapsed = time.time() - start_time
                expected_elapsed = i / queries_per_second
                if elapsed < expected_elapsed:
                    await asyncio.sleep(expected_elapsed - elapsed)

        except Exception as e:
            self.logger.error(f"Load test interrupted: {e}")

        total_time = time.time() - start_time
        end_memory = self._get_memory_usage()
        end_threads = self._get_thread_count()

        # Calculate stability metrics
        success_rate = (
            successful_queries / (successful_queries + failed_queries)
            if (successful_queries + failed_queries) > 0
            else 0
        )
        average_response_time = (
            sum(response_times) / len(response_times) if response_times else 0
        )
        memory_growth = end_memory - start_memory
        thread_growth = end_threads - start_threads

        # Stability thresholds
        acceptable_success_rate = 0.8  # 80%
        acceptable_avg_response = 5000  # 5 seconds
        acceptable_memory_growth = 50  # 50MB
        acceptable_thread_growth = 5  # 5 threads

        system_stable = (
            success_rate >= acceptable_success_rate
            and average_response_time <= acceptable_avg_response
            and memory_growth <= acceptable_memory_growth
            and thread_growth <= acceptable_thread_growth
        )

        return {
            "success": system_stable,
            "details": f"Load stability: {success_rate:.1%} success, {average_response_time:.0f}ms avg response",
            "metrics": {
                "load_duration_seconds": total_time,
                "total_queries_attempted": successful_queries + failed_queries,
                "successful_queries": successful_queries,
                "failed_queries": failed_queries,
                "success_rate": success_rate,
                "average_response_time_ms": average_response_time,
                "memory_growth_mb": memory_growth,
                "thread_growth": thread_growth,
                "queries_per_second_achieved": (
                    (successful_queries + failed_queries) / total_time
                    if total_time > 0
                    else 0
                ),
                "system_stable": system_stable,
                "response_times_sample": response_times[-10:],  # Last 10 response times
            },
        }

    # MONITORING AND LOGGING TESTS

    async def test_logging_and_monitoring_functionality(self) -> Dict[str, Any]:
        """Test logging and monitoring functionality."""
        # Create a test log handler to capture logs
        test_logs = []

        class TestLogHandler(logging.Handler):
            def emit(self, record):
                test_logs.append(self.format(record))

        # Add test handler to logger
        test_handler = TestLogHandler()
        test_handler.setLevel(logging.INFO)

        # Get the multi-agent service logger (if it exists)
        service_logger = logging.getLogger("voice_agent")
        original_level = service_logger.level
        service_logger.addHandler(test_handler)
        service_logger.setLevel(logging.INFO)

        try:
            if not self.multi_agent_service.multi_agent_enabled:
                return {
                    "success": False,
                    "details": "Multi-agent service not available",
                }

            # Perform operations that should generate logs
            test_operations = [
                "What's the weather in London?",
                "Calculate 25 * 43",
                "Invalid query with special characters: !@#$%",
                "List my files and folders",
            ]

            initial_log_count = len(test_logs)

            for operation in test_operations:
                try:
                    await self.multi_agent_service.process_message(operation)
                except Exception:
                    pass  # Expected for some test operations

            final_log_count = len(test_logs)
            logs_generated = final_log_count - initial_log_count

            # Analyze log content
            recent_logs = test_logs[initial_log_count:]

            has_info_logs = any("INFO" in log for log in recent_logs)
            has_error_logs = any("ERROR" in log or "WARN" in log for log in recent_logs)
            has_relevant_content = any(
                any(
                    term in log.lower()
                    for term in ["agent", "query", "response", "service"]
                )
                for log in recent_logs
            )

            # Check service info functionality
            service_info = self.multi_agent_service.get_service_info()
            has_service_info = isinstance(service_info, dict) and len(service_info) > 0

            logging_functional = (
                logs_generated > 0
                and (has_info_logs or has_error_logs)
                and has_relevant_content
                and has_service_info
            )

            return {
                "success": logging_functional,
                "details": f"Logging: {logs_generated} logs generated, service info available: {has_service_info}",
                "metrics": {
                    "logs_generated": logs_generated,
                    "has_info_logs": has_info_logs,
                    "has_error_logs": has_error_logs,
                    "has_relevant_content": has_relevant_content,
                    "has_service_info": has_service_info,
                    "service_info_keys": (
                        list(service_info.keys())
                        if isinstance(service_info, dict)
                        else []
                    ),
                    "recent_logs_sample": recent_logs[-5:],  # Last 5 logs
                    "logging_functional": logging_functional,
                },
            }

        finally:
            # Cleanup: remove test handler and restore original level
            service_logger.removeHandler(test_handler)
            service_logger.setLevel(original_level)


async def run_comprehensive_performance_reliability_tests():
    """Run all comprehensive performance and reliability tests."""
    print("⚡ Starting Comprehensive Performance and Reliability Test Suite")
    print("=" * 80)

    test_suite = PerformanceReliabilityTestSuite()
    framework = test_suite.framework

    try:
        await test_suite.setup()

        # Define all tests
        tests = [
            # Performance benchmarking tests
            (
                "Single Agent Response Times",
                test_suite.test_single_agent_response_times,
                TestCategory.PERFORMANCE,
                TestSeverity.HIGH,
            ),
            (
                "Concurrent Agent Operations",
                test_suite.test_concurrent_agent_operations,
                TestCategory.PERFORMANCE,
                TestSeverity.HIGH,
            ),
            # Memory and resource tests
            (
                "Memory Usage and Cleanup",
                test_suite.test_memory_usage_and_cleanup,
                TestCategory.PERFORMANCE,
                TestSeverity.MEDIUM,
            ),
            (
                "Resource Leak Detection",
                test_suite.test_resource_leak_detection,
                TestCategory.PERFORMANCE,
                TestSeverity.HIGH,
            ),
            # Reliability and resilience tests
            (
                "Error Recovery Resilience",
                test_suite.test_error_recovery_resilience,
                TestCategory.ERROR_HANDLING,
                TestSeverity.HIGH,
            ),
            (
                "System Stability Under Load",
                test_suite.test_system_stability_under_load,
                TestCategory.PERFORMANCE,
                TestSeverity.CRITICAL,
            ),
            # Monitoring and logging tests
            (
                "Logging and Monitoring Functionality",
                test_suite.test_logging_and_monitoring_functionality,
                TestCategory.FUNCTIONALITY,
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
                timeout_seconds=120.0,  # Extended timeout for performance tests
            )

        # Generate and display results
        framework.print_test_summary()
        framework.save_test_report("performance_reliability_test_report.json")

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
        results = asyncio.run(run_comprehensive_performance_reliability_tests())

        if "error" in results:
            print(f"\n💥 Test suite failed: {results['error']}")
            sys.exit(1)

        success_rate = results["summary"]["success_rate"]
        if success_rate >= 0.8:
            print(
                f"\n🎉 Performance and reliability tests completed successfully! ({success_rate:.1%} pass rate)"
            )
            sys.exit(0)
        elif success_rate >= 0.6:
            print(
                f"\n⚠️  Performance and reliability tests mostly successful ({success_rate:.1%} pass rate)"
            )
            sys.exit(1)
        else:
            print(
                f"\n❌ Performance and reliability tests failed ({success_rate:.1%} pass rate)"
            )
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        sys.exit(1)
