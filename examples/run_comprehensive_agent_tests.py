#!/usr/bin/env python3
"""
Master Test Runner for Comprehensive Agent Testing

This script runs all comprehensive agent test suites and provides:
- Unified test execution and reporting
- devenv environment validation
- Consolidated results and metrics
- Test suite dependency checking
- Performance benchmarking across all agents
- HTML report generation for comprehensive analysis
"""

import asyncio
import logging
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any


class MasterTestRunner:
    """Master test runner for all agent test suites."""

    def __init__(self, verbose: bool = False, save_reports: bool = True):
        self.verbose = verbose
        self.save_reports = save_reports
        self.logger = self._setup_logging()
        self.results = {}
        self.start_time = None
        self.end_time = None

    def _setup_logging(self) -> logging.Logger:
        """Set up comprehensive logging."""
        logger = logging.getLogger("master_test_runner")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            # File handler for detailed logs
            if self.save_reports:
                file_handler = logging.FileHandler("comprehensive_agent_tests.log")
                file_formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)

        return logger

    def validate_devenv_environment(self) -> Dict[str, Any]:
        """Validate that we're running in the correct devenv environment."""
        self.logger.info("Validating devenv environment...")

        validation_results = {
            "python_version": sys.version,
            "python_path": sys.executable,
            "current_directory": str(Path.cwd()),
            "src_directory_exists": Path("src").exists(),
            "voice_agent_module_exists": Path("src/voice_agent").exists(),
            "config_files_exist": Path("src/voice_agent/config/default.yaml").exists(),
            "devenv_indicators": [],
        }

        # Check for devenv indicators
        if "devenv" in str(Path.cwd()) or "devenv" in sys.executable:
            validation_results["devenv_indicators"].append("devenv in path")

        # Check for nix store paths
        if "/nix/store" in sys.executable:
            validation_results["devenv_indicators"].append("nix store python")

        # Try importing key modules
        try:
            # Test import without storing unused reference
            __import__("voice_agent.core.config")
            validation_results["voice_agent_import"] = True
        except ImportError as e:
            validation_results["voice_agent_import"] = False
            validation_results["import_error"] = str(e)

        # Check if running in correct directory
        expected_files = [
            "devenv.nix",
            "devenv.lock",
            "src/voice_agent/main.py",
            "src/voice_agent/config/default.yaml",
        ]

        missing_files = []
        for file_path in expected_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)

        validation_results["missing_files"] = missing_files
        validation_results["environment_valid"] = (
            validation_results["voice_agent_import"]
            and len(missing_files) == 0
            and validation_results["src_directory_exists"]
        )

        if validation_results["environment_valid"]:
            self.logger.info("✅ devenv environment validation passed")
        else:
            self.logger.warning("⚠️  devenv environment validation failed")
            self.logger.warning(f"Missing files: {missing_files}")

        return validation_results

    async def run_test_suite(self, suite_name: str, test_function) -> Dict[str, Any]:
        """Run a single test suite with error handling."""
        self.logger.info(f"🚀 Starting {suite_name}...")

        start_time = time.time()

        try:
            results = await test_function()
            duration = time.time() - start_time

            if isinstance(results, dict) and "error" not in results:
                self.logger.info(
                    f"✅ {suite_name} completed successfully in {duration:.1f}s"
                )

                # Add duration to results
                if "metrics" not in results:
                    results["metrics"] = {}
                results["metrics"]["suite_duration_seconds"] = duration

                return {
                    "success": True,
                    "results": results,
                    "duration": duration,
                    "error": None,
                }
            else:
                error_msg = (
                    results.get("error", "Unknown error")
                    if isinstance(results, dict)
                    else "Invalid results format"
                )
                self.logger.error(f"❌ {suite_name} failed: {error_msg}")

                return {
                    "success": False,
                    "results": results,
                    "duration": duration,
                    "error": error_msg,
                }

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.logger.error(f"💥 {suite_name} crashed: {error_msg}")

            return {
                "success": False,
                "results": None,
                "duration": duration,
                "error": error_msg,
            }

    async def run_all_test_suites(self) -> Dict[str, Any]:
        """Run all comprehensive test suites."""
        self.start_time = time.time()

        # Define test suites in execution order
        test_suites = [
            ("Agent Testing Framework", self._test_framework_functionality),
            ("InformationAgent Tests", run_comprehensive_information_agent_tests),
            ("UtilityAgent Tests", run_comprehensive_utility_agent_tests),
            ("ProductivityAgent Tests", run_comprehensive_productivity_agent_tests),
            ("Agent Integration Tests", run_comprehensive_agent_integration_tests),
            (
                "Performance & Reliability Tests",
                run_comprehensive_performance_reliability_tests,
            ),
        ]

        suite_results = {}

        # Run each test suite
        for suite_name, test_function in test_suites:
            suite_result = await self.run_test_suite(suite_name, test_function)
            suite_results[suite_name] = suite_result

            # Break early if critical tests fail
            if not suite_result["success"] and "Framework" in suite_name:
                self.logger.critical(
                    "Framework tests failed - aborting remaining tests"
                )
                break

        self.end_time = time.time()

        # Compile overall results
        overall_results = self._compile_overall_results(suite_results)

        return overall_results

    async def _test_framework_functionality(self) -> Dict[str, Any]:
        """Test the framework itself."""
        framework = AgentTestFramework()

        try:
            # Test framework setup and cleanup
            test_env = await framework.setup_test_environment()

            # Validate test environment
            validation_checks = {
                "config_loaded": test_env.get("config") is not None,
                "tool_executor_available": test_env.get("tool_executor") is not None,
                "temp_dir_created": test_env.get("temp_dir") is not None
                and Path(test_env["temp_dir"]).exists(),
            }

            await framework.cleanup_test_environment()

            all_passed = all(validation_checks.values())

            return {
                "summary": {
                    "total_tests": len(validation_checks),
                    "passed": sum(validation_checks.values()),
                    "failed": len(validation_checks) - sum(validation_checks.values()),
                    "success_rate": sum(validation_checks.values())
                    / len(validation_checks),
                },
                "validation_checks": validation_checks,
                "framework_functional": all_passed,
            }

        except Exception as e:
            return {
                "error": f"Framework test failed: {str(e)}",
                "framework_functional": False,
            }

    def _compile_overall_results(self, suite_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile overall test results and metrics."""
        total_duration = (
            self.end_time - self.start_time if self.start_time and self.end_time else 0
        )

        # Count overall statistics
        total_suites = len(suite_results)
        successful_suites = sum(
            1 for result in suite_results.values() if result["success"]
        )
        failed_suites = total_suites - successful_suites

        # Aggregate test statistics from all suites
        total_tests = 0
        total_passed = 0
        total_failed = 0

        suite_summaries = {}

        for suite_name, suite_result in suite_results.items():
            if suite_result["success"] and suite_result["results"]:
                results = suite_result["results"]

                if isinstance(results, dict) and "summary" in results:
                    summary = results["summary"]
                    suite_total = summary.get("total_tests", 0)
                    suite_passed = summary.get("passed", 0)
                    suite_failed = summary.get("failed", 0)

                    total_tests += suite_total
                    total_passed += suite_passed
                    total_failed += suite_failed

                    suite_summaries[suite_name] = {
                        "total_tests": suite_total,
                        "passed": suite_passed,
                        "failed": suite_failed,
                        "success_rate": summary.get("success_rate", 0),
                        "duration": suite_result["duration"],
                    }
                else:
                    suite_summaries[suite_name] = {
                        "total_tests": 0,
                        "passed": 0,
                        "failed": 1,  # Suite failure
                        "success_rate": 0.0,
                        "duration": suite_result["duration"],
                        "error": suite_result.get("error"),
                    }
            else:
                suite_summaries[suite_name] = {
                    "total_tests": 0,
                    "passed": 0,
                    "failed": 1,  # Suite failure
                    "success_rate": 0.0,
                    "duration": suite_result["duration"],
                    "error": suite_result.get("error"),
                }

        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        suite_success_rate = successful_suites / total_suites if total_suites > 0 else 0

        return {
            "execution_summary": {
                "total_duration_seconds": total_duration,
                "total_suites": total_suites,
                "successful_suites": successful_suites,
                "failed_suites": failed_suites,
                "suite_success_rate": suite_success_rate,
            },
            "test_summary": {
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "overall_success_rate": overall_success_rate,
            },
            "suite_details": suite_summaries,
            "full_results": suite_results,
            "timestamp": time.time(),
            "environment": "devenv",
        }

    def generate_html_report(
        self,
        results: Dict[str, Any],
        output_path: str = "comprehensive_agent_test_report.html",
    ):
        """Generate an HTML report of all test results."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Agent Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; border-left: 4px solid #007bff; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        .suite-results {{ margin-bottom: 30px; }}
        .suite-card {{ background: white; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 15px; }}
        .suite-header {{ background: #f8f9fa; padding: 15px; border-bottom: 1px solid #ddd; cursor: pointer; }}
        .suite-content {{ padding: 15px; display: none; }}
        .success {{ color: #28a745; }}
        .failure {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        .progress-bar {{ width: 100%; height: 20px; background-color: #e9ecef; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background-color: #28a745; transition: width 0.3s ease; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; }}
        .expandable {{ cursor: pointer; }}
        .expandable:hover {{ background-color: #f8f9fa; }}
    </style>
    <script>
        function toggleSuite(suiteId) {{
            const content = document.getElementById(suiteId);
            content.style.display = content.style.display === 'none' ? 'block' : 'none';
        }}
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Comprehensive Agent Test Report</h1>
            <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results.get('timestamp', time.time())))}</p>
        </div>
        
        <div class="summary">
            <div class="metric-card">
                <div class="metric-value">{results['execution_summary']['total_suites']}</div>
                <div class="metric-label">Test Suites</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{results['test_summary']['total_tests']}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'success' if results['test_summary']['overall_success_rate'] >= 0.8 else 'warning' if results['test_summary']['overall_success_rate'] >= 0.6 else 'failure'}">{results['test_summary']['overall_success_rate']:.1%}</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{results['execution_summary']['total_duration_seconds']:.1f}s</div>
                <div class="metric-label">Total Duration</div>
            </div>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" style="width: {results['test_summary']['overall_success_rate'] * 100:.1f}%"></div>
        </div>
        
        <div class="suite-results">
            <h2>Test Suite Results</h2>"""

        for suite_name, suite_data in results["suite_details"].items():
            success_class = (
                "success"
                if suite_data["success_rate"] >= 0.8
                else "warning" if suite_data["success_rate"] >= 0.6 else "failure"
            )
            suite_id = suite_name.replace(" ", "_").replace("&", "and").lower()

            html_content += f"""
            <div class="suite-card">
                <div class="suite-header expandable" onclick="toggleSuite('{suite_id}_content')">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3>{suite_name}</h3>
                        <span class="{success_class}">
                            {suite_data['passed']}/{suite_data['total_tests']} tests passed 
                            ({suite_data['success_rate']:.1%})
                        </span>
                    </div>
                    <div style="margin-top: 10px;">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {suite_data['success_rate'] * 100:.1f}%"></div>
                        </div>
                    </div>
                </div>
                <div id="{suite_id}_content" class="suite-content">
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Duration</td><td>{suite_data['duration']:.2f}s</td></tr>
                        <tr><td>Total Tests</td><td>{suite_data['total_tests']}</td></tr>
                        <tr><td>Passed</td><td class="success">{suite_data['passed']}</td></tr>
                        <tr><td>Failed</td><td class="failure">{suite_data['failed']}</td></tr>
                        <tr><td>Success Rate</td><td class="{success_class}">{suite_data['success_rate']:.1%}</td></tr>
                    </table>
                    {'<p class="failure">Error: ' + suite_data.get('error', '') + '</p>' if suite_data.get('error') else ''}
                </div>
            </div>"""

        html_content += """
        </div>
        
        <div class="footer" style="margin-top: 40px; text-align: center; color: #666; border-top: 1px solid #ddd; padding-top: 20px;">
            <p>Generated by Comprehensive Agent Test Framework</p>
            <p>Environment: devenv | Voice Agent Multi-Agent System</p>
        </div>
    </div>
</body>
</html>"""

        with open(output_path, "w") as f:
            f.write(html_content)

        self.logger.info(f"📊 HTML report generated: {output_path}")

    def print_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of all test results."""
        print("\n" + "=" * 100)
        print("🤖 COMPREHENSIVE AGENT TEST RESULTS SUMMARY")
        print("=" * 100)

        exec_summary = results["execution_summary"]
        test_summary = results["test_summary"]

        print(
            f"⏱️  Total Execution Time: {exec_summary['total_duration_seconds']:.1f} seconds"
        )
        print(
            f"📦 Test Suites: {exec_summary['successful_suites']}/{exec_summary['total_suites']} successful ({exec_summary['suite_success_rate']:.1%})"
        )
        print(
            f"🧪 Individual Tests: {test_summary['total_passed']}/{test_summary['total_tests']} passed ({test_summary['overall_success_rate']:.1%})"
        )

        print(f"\n📋 Suite Breakdown:")
        for suite_name, suite_data in results["suite_details"].items():
            status_icon = (
                "✅"
                if suite_data["success_rate"] >= 0.8
                else "⚠️" if suite_data["success_rate"] >= 0.6 else "❌"
            )
            print(
                f"  {status_icon} {suite_name:35} {suite_data['passed']:3}/{suite_data['total_tests']:3} ({suite_data['success_rate']:6.1%}) [{suite_data['duration']:6.1f}s]"
            )
            if suite_data.get("error"):
                print(f"      Error: {suite_data['error']}")

        print(f"\n🎯 Overall Assessment:")
        if test_summary["overall_success_rate"] >= 0.9:
            print("🎉 EXCELLENT: All agent systems are working exceptionally well!")
        elif test_summary["overall_success_rate"] >= 0.8:
            print("✅ GOOD: Agent systems are working well with minor issues.")
        elif test_summary["overall_success_rate"] >= 0.6:
            print(
                "⚠️  ACCEPTABLE: Agent systems are mostly functional but need attention."
            )
        else:
            print("❌ NEEDS WORK: Significant issues found in agent systems.")

        print("=" * 100)


async def main():
    """Main entry point for comprehensive agent testing."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive tests for all agent systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s                     # Run all tests with default settings
  %(prog)s --verbose           # Run with verbose logging
  %(prog)s --no-reports        # Skip saving detailed reports
  %(prog)s --quick             # Run quick validation only
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--no-reports", action="store_true", help="Skip saving detailed test reports"
    )

    parser.add_argument(
        "--quick", action="store_true", help="Run quick environment validation only"
    )

    args = parser.parse_args()

    # Create master test runner
    runner = MasterTestRunner(verbose=args.verbose, save_reports=not args.no_reports)

    try:
        print("🚀 Starting Comprehensive Agent Test Suite")
        print("=" * 80)

        # Validate environment first
        env_validation = runner.validate_devenv_environment()

        if not env_validation["environment_valid"]:
            print("❌ Environment validation failed!")
            print("Please ensure you're running in the devenv environment:")
            print("  devenv shell")
            print(f"Missing files: {env_validation['missing_files']}")
            return 1

        if args.quick:
            print("✅ Quick environment validation passed!")
            return 0

        # Run all test suites
        results = await runner.run_all_test_suites()

        # Generate reports
        if runner.save_reports:
            # Save JSON report
            with open("comprehensive_agent_test_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

            # Generate HTML report
            runner.generate_html_report(results)

        # Print summary
        runner.print_summary(results)

        # Determine exit code
        overall_success_rate = results["test_summary"]["overall_success_rate"]
        if overall_success_rate >= 0.8:
            print(f"\n🎉 SUCCESS: Comprehensive agent testing completed successfully!")
            return 0
        elif overall_success_rate >= 0.6:
            print(f"\n⚠️  PARTIAL SUCCESS: Most tests passed but some issues found.")
            return 1
        else:
            print(f"\n❌ FAILURE: Significant test failures detected.")
            return 2

    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        runner.logger.exception("Test execution failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
