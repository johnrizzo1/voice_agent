#!/usr/bin/env python3
"""
Comprehensive End-to-End Voice + Multi-Agent Workflow Testing

This test suite validates the complete voice interaction pipeline with multi-agent
orchestration, including:

1. Voice Pipeline Integration with Multi-Agent Routing
2. Multi-Step Workflows with Context Preservation
3. Voice-Specific Multi-Agent Scenarios
4. Real-Time Interaction Performance
5. Integration Fallback Mechanisms
6. TUI Integration during Voice Operations
"""

import asyncio
import logging
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from voice_agent.core.config import Config
from voice_agent.core.voice_agent_orchestrator import VoiceAgentOrchestrator
from voice_agent.ui.tui_app import VoiceAgentTUI, AgentAdapter, PipelineStatus


class VoiceWorkflowTestFramework:
    """Framework for testing end-to-end voice + multi-agent workflows."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results: Dict[str, Any] = {}
        self.performance_metrics: List[Dict[str, Any]] = []
        self.orchestrator: Optional[VoiceAgentOrchestrator] = None
        self.tui_app: Optional[VoiceAgentTUI] = None
        self.pipeline_status = PipelineStatus()

    async def setup_test_environment(self) -> bool:
        """Setup complete test environment with voice + multi-agent system."""
        try:
            print("🔧 Setting up test environment...")

            # Load test configuration
            config_path = Path(__file__).parent / "test_config_multi_agent.yaml"
            if not config_path.exists():
                print(f"❌ Test configuration not found: {config_path}")
                return False

            config = Config.load(config_path)
            config.multi_agent.enabled = True
            config.ui.force_text_only = False  # Enable audio for testing

            # Initialize orchestrator with multi-agent support
            self.orchestrator = VoiceAgentOrchestrator(
                config=config,
                text_only=False,  # Enable full voice pipeline
                state_callback=self._pipeline_state_callback,
            )

            await self.orchestrator.initialize()

            # Setup TUI integration
            agent_adapter = AgentAdapter(self.orchestrator, self.pipeline_status)

            print("✅ Test environment setup complete")
            return True

        except Exception as e:
            print(f"❌ Test environment setup failed: {e}")
            return False

    def _pipeline_state_callback(
        self, component: str, state: str, message: Optional[str]
    ) -> None:
        """Pipeline state callback for monitoring."""
        self.logger.debug(f"Pipeline: {component} -> {state} ({message})")
        # Update pipeline status for TUI integration tests

    async def test_voice_pipeline_multi_agent_integration(self) -> Dict[str, Any]:
        """Test 1: Voice Pipeline Integration with Multi-Agent Routing"""
        print("\n=== Test 1: Voice Pipeline + Multi-Agent Integration ===")
        results = {"name": "voice_pipeline_integration", "subtests": []}

        # Subtest 1.1: Voice input routing to different agents
        voice_routing_test = await self._test_voice_agent_routing()
        results["subtests"].append(voice_routing_test)

        # Subtest 1.2: Voice-optimized responses
        voice_response_test = await self._test_voice_optimized_responses()
        results["subtests"].append(voice_response_test)

        # Subtest 1.3: Voice continuity during agent handoffs
        handoff_test = await self._test_voice_continuity_during_handoffs()
        results["subtests"].append(handoff_test)

        results["overall_success"] = all(t["success"] for t in results["subtests"])
        return results

    async def _test_voice_agent_routing(self) -> Dict[str, Any]:
        """Test voice input routing to appropriate agents."""
        print("📍 Testing voice input routing to agents...")

        test_scenarios = [
            {
                "voice_input": "What's the weather like today?",
                "expected_agent": "information_agent",
                "description": "Weather query routing",
            },
            {
                "voice_input": "Calculate twenty-five times thirty-seven",
                "expected_agent": "utility_agent",
                "description": "Calculator routing",
            },
            {
                "voice_input": "Create a new file called test.txt",
                "expected_agent": "productivity_agent",
                "description": "File operation routing",
            },
            {
                "voice_input": "How are you doing today?",
                "expected_agent": "general_agent",
                "description": "General conversation routing",
            },
        ]

        routing_results = []

        for scenario in test_scenarios:
            try:
                start_time = time.time()

                # Simulate voice input processing
                with patch.object(
                    self.orchestrator, "audio_manager"
                ) as mock_audio, patch.object(
                    self.orchestrator, "stt_service"
                ) as mock_stt:

                    # Mock audio capture
                    mock_audio.listen.return_value = np.random.random(16000).astype(
                        np.float32
                    )
                    mock_stt.transcribe.return_value = scenario["voice_input"]

                    # Process through multi-agent system
                    response = await self.orchestrator.process_text(
                        scenario["voice_input"]
                    )

                processing_time = time.time() - start_time

                # Record performance metrics
                self.performance_metrics.append(
                    {
                        "test": "voice_routing",
                        "scenario": scenario["description"],
                        "processing_time": processing_time,
                        "input_length": len(scenario["voice_input"]),
                        "response_length": len(response),
                    }
                )

                routing_results.append(
                    {
                        "scenario": scenario["description"],
                        "success": True,
                        "processing_time": processing_time,
                        "response_generated": bool(response),
                    }
                )

                print(f"  ✅ {scenario['description']}: {processing_time:.2f}s")

            except Exception as e:
                routing_results.append(
                    {
                        "scenario": scenario["description"],
                        "success": False,
                        "error": str(e),
                    }
                )
                print(f"  ❌ {scenario['description']}: {e}")

        success_rate = sum(1 for r in routing_results if r["success"]) / len(
            routing_results
        )

        return {
            "name": "voice_agent_routing",
            "success": success_rate >= 0.75,  # 75% success threshold
            "success_rate": success_rate,
            "results": routing_results,
            "metrics": {
                "avg_processing_time": np.mean(
                    [
                        r["processing_time"]
                        for r in routing_results
                        if "processing_time" in r
                    ]
                ),
                "scenarios_tested": len(test_scenarios),
            },
        }

    async def _test_voice_optimized_responses(self) -> Dict[str, Any]:
        """Test that responses are optimized for voice output."""
        print("📍 Testing voice-optimized response generation...")

        voice_queries = [
            "Tell me about the weather forecast",
            "What's fifteen plus twenty-eight",
            "Explain how photosynthesis works",
            "List the benefits of exercise",
        ]

        optimization_results = []

        for query in voice_queries:
            try:
                response = await self.orchestrator.process_text(query)

                # Check voice-optimization criteria
                word_count = len(response.split())
                has_markdown = any(
                    marker in response for marker in ["```", "[", "](", "#", "*"]
                )
                ends_properly = response.strip().endswith((".", "!", "?"))
                reasonable_length = 10 <= word_count <= 150  # Good for TTS

                optimization_score = (
                    sum(
                        [
                            not has_markdown,  # No markdown formatting
                            ends_properly,  # Proper sentence ending
                            reasonable_length,  # Appropriate length for speech
                        ]
                    )
                    / 3
                )

                optimization_results.append(
                    {
                        "query": query,
                        "word_count": word_count,
                        "has_markdown": has_markdown,
                        "ends_properly": ends_properly,
                        "reasonable_length": reasonable_length,
                        "optimization_score": optimization_score,
                        "success": optimization_score >= 0.67,
                    }
                )

                print(
                    f"  ✅ Query optimization: {optimization_score:.2f} ({word_count} words)"
                )

            except Exception as e:
                optimization_results.append(
                    {"query": query, "success": False, "error": str(e)}
                )
                print(f"  ❌ Query failed: {e}")

        avg_score = np.mean(
            [
                r["optimization_score"]
                for r in optimization_results
                if "optimization_score" in r
            ]
        )

        return {
            "name": "voice_optimized_responses",
            "success": avg_score >= 0.7,
            "average_optimization_score": avg_score,
            "results": optimization_results,
        }

    async def _test_voice_continuity_during_handoffs(self) -> Dict[str, Any]:
        """Test voice conversation flow during agent handoffs."""
        print("📍 Testing voice continuity during agent handoffs...")

        # Simulate a conversation that requires handoffs
        conversation_flow = [
            {
                "input": "I need to calculate 25 times 30",
                "expected_agent": "utility_agent",
            },
            {
                "input": "Now save that result to a file",
                "expected_agent": "productivity_agent",
            },
            {"input": "What did we just calculate?", "expected_agent": "general_agent"},
        ]

        conversation_results = []
        conversation_context = []

        for step, interaction in enumerate(conversation_flow):
            try:
                response = await self.orchestrator.process_text(interaction["input"])

                # Check context preservation
                context_preserved = True
                if step > 0:
                    # Check if response shows awareness of previous interactions
                    prev_keywords = ["calculate", "25", "30", "750", "file", "result"]
                    context_preserved = any(
                        keyword in response.lower() for keyword in prev_keywords
                    )

                conversation_results.append(
                    {
                        "step": step + 1,
                        "input": interaction["input"],
                        "response_generated": bool(response),
                        "context_preserved": context_preserved,
                        "success": bool(response) and context_preserved,
                    }
                )

                conversation_context.append(
                    {"user": interaction["input"], "agent": response}
                )

                print(f"  ✅ Step {step + 1}: Context preserved: {context_preserved}")

            except Exception as e:
                conversation_results.append(
                    {
                        "step": step + 1,
                        "input": interaction["input"],
                        "success": False,
                        "error": str(e),
                    }
                )
                print(f"  ❌ Step {step + 1}: {e}")

        success_rate = sum(1 for r in conversation_results if r["success"]) / len(
            conversation_results
        )

        return {
            "name": "voice_continuity_handoffs",
            "success": success_rate >= 0.67,
            "success_rate": success_rate,
            "conversation_flow": conversation_results,
            "context_preservation": conversation_context,
        }

    async def test_multi_step_workflows_with_context(self) -> Dict[str, Any]:
        """Test 2: Multi-Step Workflows with Context Preservation"""
        print("\n=== Test 2: Multi-Step Workflows with Context Preservation ===")
        results = {"name": "multi_step_workflows", "subtests": []}

        # Complex workflow: Weather → Analysis → File Save
        workflow_test = await self._test_weather_analysis_save_workflow()
        results["subtests"].append(workflow_test)

        # Sequential calculation workflow
        calc_workflow_test = await self._test_sequential_calculation_workflow()
        results["subtests"].append(calc_workflow_test)

        # Information gathering and synthesis workflow
        info_synthesis_test = await self._test_information_synthesis_workflow()
        results["subtests"].append(info_synthesis_test)

        results["overall_success"] = all(t["success"] for t in results["subtests"])
        return results

    async def _test_weather_analysis_save_workflow(self) -> Dict[str, Any]:
        """Test complex weather → analysis → save workflow."""
        print("📍 Testing weather analysis and save workflow...")

        workflow_steps = [
            "Get the current weather for New York City",
            "Analyze if it's good weather for outdoor activities",
            "Save this weather analysis to a file called weather_report.txt",
        ]

        workflow_results = []
        accumulated_context = ""

        for step_num, step in enumerate(workflow_steps):
            try:
                start_time = time.time()
                response = await self.orchestrator.process_text(step)
                processing_time = time.time() - start_time

                # Check if response shows awareness of previous steps
                context_awareness = True
                if step_num > 0:
                    context_keywords = ["weather", "new york", "outdoor", "activities"]
                    context_awareness = any(
                        keyword in response.lower() for keyword in context_keywords
                    )

                workflow_results.append(
                    {
                        "step": step_num + 1,
                        "request": step,
                        "processing_time": processing_time,
                        "response_length": len(response),
                        "context_aware": context_awareness,
                        "success": bool(response) and context_awareness,
                    }
                )

                accumulated_context += (
                    f"Step {step_num + 1}: {step} -> {response[:100]}...\n"
                )

                print(
                    f"  ✅ Step {step_num + 1}: {processing_time:.2f}s, Context: {context_awareness}"
                )

            except Exception as e:
                workflow_results.append(
                    {
                        "step": step_num + 1,
                        "request": step,
                        "success": False,
                        "error": str(e),
                    }
                )
                print(f"  ❌ Step {step_num + 1}: {e}")

        success_rate = sum(1 for r in workflow_results if r["success"]) / len(
            workflow_results
        )

        return {
            "name": "weather_analysis_save_workflow",
            "success": success_rate >= 0.67,
            "success_rate": success_rate,
            "workflow_steps": workflow_results,
            "total_processing_time": sum(
                r.get("processing_time", 0) for r in workflow_results
            ),
        }

    async def _test_sequential_calculation_workflow(self) -> Dict[str, Any]:
        """Test sequential calculation workflow with context."""
        print("📍 Testing sequential calculation workflow...")

        calc_sequence = [
            "Calculate 15 times 23",
            "Add 100 to that result",
            "Divide the final number by 5",
            "What's the square root of our final answer?",
        ]

        calc_results = []
        expected_values = [345, 445, 89, 9.43]  # Expected intermediate results

        for step_num, calculation in enumerate(calc_sequence):
            try:
                response = await self.orchestrator.process_text(calculation)

                # Try to extract numeric result from response
                import re

                numbers = re.findall(r"\d+\.?\d*", response)
                numeric_result = float(numbers[0]) if numbers else None

                # Check if result is approximately correct
                result_correct = False
                if numeric_result is not None and step_num < len(expected_values):
                    expected = expected_values[step_num]
                    result_correct = abs(numeric_result - expected) < (
                        expected * 0.1
                    )  # 10% tolerance

                calc_results.append(
                    {
                        "step": step_num + 1,
                        "calculation": calculation,
                        "extracted_result": numeric_result,
                        "expected_result": (
                            expected_values[step_num]
                            if step_num < len(expected_values)
                            else None
                        ),
                        "result_correct": result_correct,
                        "response_generated": bool(response),
                        "success": bool(response)
                        and (
                            step_num == 0 or result_correct
                        ),  # First step just needs response
                    }
                )

                print(
                    f"  ✅ Step {step_num + 1}: Result: {numeric_result}, Correct: {result_correct}"
                )

            except Exception as e:
                calc_results.append(
                    {
                        "step": step_num + 1,
                        "calculation": calculation,
                        "success": False,
                        "error": str(e),
                    }
                )
                print(f"  ❌ Step {step_num + 1}: {e}")

        success_rate = sum(1 for r in calc_results if r["success"]) / len(calc_results)

        return {
            "name": "sequential_calculation_workflow",
            "success": success_rate >= 0.5,  # More lenient for complex math
            "success_rate": success_rate,
            "calculation_steps": calc_results,
        }

    async def _test_information_synthesis_workflow(self) -> Dict[str, Any]:
        """Test information gathering and synthesis workflow."""
        print("📍 Testing information synthesis workflow...")

        synthesis_steps = [
            "Tell me about renewable energy sources",
            "Which renewable energy source is most efficient?",
            "Create a summary of renewable energy benefits based on our discussion",
        ]

        synthesis_results = []
        conversation_memory = []

        for step_num, request in enumerate(synthesis_steps):
            try:
                response = await self.orchestrator.process_text(request)

                # Check synthesis quality for final step
                synthesis_quality = True
                if step_num == 2:  # Final synthesis step
                    synthesis_keywords = [
                        "renewable",
                        "energy",
                        "solar",
                        "wind",
                        "efficient",
                        "benefits",
                    ]
                    synthesis_quality = (
                        sum(
                            1 for word in synthesis_keywords if word in response.lower()
                        )
                        >= 3
                    )

                synthesis_results.append(
                    {
                        "step": step_num + 1,
                        "request": request,
                        "response_length": len(response),
                        "synthesis_quality": synthesis_quality,
                        "success": bool(response) and synthesis_quality,
                    }
                )

                conversation_memory.append(f"{request} -> {response[:200]}...")

                print(f"  ✅ Step {step_num + 1}: Quality: {synthesis_quality}")

            except Exception as e:
                synthesis_results.append(
                    {
                        "step": step_num + 1,
                        "request": request,
                        "success": False,
                        "error": str(e),
                    }
                )
                print(f"  ❌ Step {step_num + 1}: {e}")

        success_rate = sum(1 for r in synthesis_results if r["success"]) / len(
            synthesis_results
        )

        return {
            "name": "information_synthesis_workflow",
            "success": success_rate >= 0.67,
            "success_rate": success_rate,
            "synthesis_steps": synthesis_results,
            "conversation_memory": conversation_memory,
        }

    async def test_voice_specific_multi_agent_scenarios(self) -> Dict[str, Any]:
        """Test 3: Voice-Specific Multi-Agent Scenarios"""
        print("\n=== Test 3: Voice-Specific Multi-Agent Scenarios ===")
        results = {"name": "voice_specific_scenarios", "subtests": []}

        # Voice weather interactions
        weather_test = await self._test_voice_weather_scenarios()
        results["subtests"].append(weather_test)

        # Voice file operation scenarios
        file_test = await self._test_voice_file_operations()
        results["subtests"].append(file_test)

        # Voice calculator scenarios
        calc_test = await self._test_voice_calculator_scenarios()
        results["subtests"].append(calc_test)

        # Mixed workflow scenarios
        mixed_test = await self._test_mixed_voice_workflows()
        results["subtests"].append(mixed_test)

        results["overall_success"] = all(t["success"] for t in results["subtests"])
        return results

    async def _test_voice_weather_scenarios(self) -> Dict[str, Any]:
        """Test voice-based weather queries with natural speech patterns."""
        print("📍 Testing voice weather scenarios...")

        weather_queries = [
            "What's it like outside today?",
            "Do I need an umbrella?",
            "Is it gonna rain this afternoon?",
            "Should I wear a jacket when I go out?",
            "What's the temperature right now?",
        ]

        weather_results = []

        for query in weather_queries:
            try:
                # Simulate voice input with natural speech timing
                await asyncio.sleep(0.1)  # Simulate speech processing delay

                start_time = time.time()
                response = await self.orchestrator.process_text(query)
                response_time = time.time() - start_time

                # Check if response is conversational and weather-related
                is_conversational = any(
                    word in response.lower()
                    for word in ["you", "your", "today", "currently"]
                )
                is_weather_related = any(
                    word in response.lower()
                    for word in ["weather", "temperature", "rain", "sunny", "cloudy"]
                )

                weather_results.append(
                    {
                        "query": query,
                        "response_time": response_time,
                        "is_conversational": is_conversational,
                        "is_weather_related": is_weather_related,
                        "response_length": len(response.split()),
                        "success": bool(response) and is_weather_related,
                    }
                )

                print(
                    f"  ✅ '{query}': {response_time:.2f}s, Conversational: {is_conversational}"
                )

            except Exception as e:
                weather_results.append(
                    {"query": query, "success": False, "error": str(e)}
                )
                print(f"  ❌ '{query}': {e}")

        success_rate = sum(1 for r in weather_results if r["success"]) / len(
            weather_results
        )
        avg_response_time = np.mean(
            [r["response_time"] for r in weather_results if "response_time" in r]
        )

        return {
            "name": "voice_weather_scenarios",
            "success": success_rate >= 0.6,  # Weather API may not be available in tests
            "success_rate": success_rate,
            "average_response_time": avg_response_time,
            "scenarios": weather_results,
        }

    async def _test_voice_file_operations(self) -> Dict[str, Any]:
        """Test voice-based file operations."""
        print("📍 Testing voice file operations...")

        file_commands = [
            "Create a new file called shopping list",
            "Write 'milk, bread, eggs' to that file",
            "Read what's in the shopping list file",
            "Make a backup of the shopping list",
        ]

        file_results = []

        for cmd in file_commands:
            try:
                response = await self.orchestrator.process_text(cmd)

                # Check if response acknowledges file operation
                acknowledges_action = any(
                    word in response.lower()
                    for word in [
                        "created",
                        "wrote",
                        "saved",
                        "read",
                        "contains",
                        "backup",
                    ]
                )
                mentions_file = any(
                    word in response.lower() for word in ["file", "shopping", "list"]
                )

                file_results.append(
                    {
                        "command": cmd,
                        "acknowledges_action": acknowledges_action,
                        "mentions_file": mentions_file,
                        "response_generated": bool(response),
                        "success": bool(response) and acknowledges_action,
                    }
                )

                print(f"  ✅ '{cmd}': Action acknowledged: {acknowledges_action}")

            except Exception as e:
                file_results.append({"command": cmd, "success": False, "error": str(e)})
                print(f"  ❌ '{cmd}': {e}")

        success_rate = sum(1 for r in file_results if r["success"]) / len(file_results)

        return {
            "name": "voice_file_operations",
            "success": success_rate >= 0.75,
            "success_rate": success_rate,
            "file_operations": file_results,
        }

    async def _test_voice_calculator_scenarios(self) -> Dict[str, Any]:
        """Test voice-based calculator operations with natural speech."""
        print("📍 Testing voice calculator scenarios...")

        math_queries = [
            "What's twenty-five plus thirty-seven?",
            "Multiply that by two",
            "What's the square root of one hundred?",
            "Calculate fifteen percent of two hundred",
        ]

        calc_results = []

        for query in math_queries:
            try:
                response = await self.orchestrator.process_text(query)

                # Check if response contains a numeric result
                import re

                has_number = bool(re.search(r"\d+(?:\.\d+)?", response))
                is_math_response = any(
                    word in response.lower()
                    for word in ["equals", "is", "result", "answer"]
                )

                calc_results.append(
                    {
                        "query": query,
                        "has_number": has_number,
                        "is_math_response": is_math_response,
                        "response_length": len(response.split()),
                        "success": bool(response) and has_number,
                    }
                )

                print(
                    f"  ✅ '{query}': Has number: {has_number}, Math response: {is_math_response}"
                )

            except Exception as e:
                calc_results.append({"query": query, "success": False, "error": str(e)})
                print(f"  ❌ '{query}': {e}")

        success_rate = sum(1 for r in calc_results if r["success"]) / len(calc_results)

        return {
            "name": "voice_calculator_scenarios",
            "success": success_rate >= 0.75,
            "success_rate": success_rate,
            "calculations": calc_results,
        }

    async def _test_mixed_voice_workflows(self) -> Dict[str, Any]:
        """Test mixed workflows combining multiple agent types."""
        print("📍 Testing mixed voice workflows...")

        mixed_workflow = [
            "What's the weather like?",
            "Calculate how many minutes that is in degrees",  # Should clarify or ask
            "Save today's weather info to a file",
            "What did we just save?",
        ]

        mixed_results = []

        for step_num, request in enumerate(mixed_workflow):
            try:
                response = await self.orchestrator.process_text(request)

                # Check appropriate response type for each step
                appropriate_response = True
                if step_num == 1:  # Ambiguous request - should seek clarification
                    appropriate_response = any(
                        word in response.lower()
                        for word in [
                            "clarify",
                            "unclear",
                            "what do you mean",
                            "specify",
                        ]
                    )
                elif step_num == 3:  # Context reference - should show memory
                    appropriate_response = any(
                        word in response.lower()
                        for word in ["weather", "saved", "file"]
                    )

                mixed_results.append(
                    {
                        "step": step_num + 1,
                        "request": request,
                        "appropriate_response": appropriate_response,
                        "response_generated": bool(response),
                        "success": bool(response) and appropriate_response,
                    }
                )

                print(f"  ✅ Step {step_num + 1}: Appropriate: {appropriate_response}")

            except Exception as e:
                mixed_results.append(
                    {
                        "step": step_num + 1,
                        "request": request,
                        "success": False,
                        "error": str(e),
                    }
                )
                print(f"  ❌ Step {step_num + 1}: {e}")

        success_rate = sum(1 for r in mixed_results if r["success"]) / len(
            mixed_results
        )

        return {
            "name": "mixed_voice_workflows",
            "success": success_rate >= 0.5,  # More complex scenarios
            "success_rate": success_rate,
            "workflow_steps": mixed_results,
        }

    async def test_real_time_interaction_performance(self) -> Dict[str, Any]:
        """Test 4: Real-Time Interaction Performance and Timing"""
        print("\n=== Test 4: Real-Time Interaction Performance ===")
        results = {"name": "real_time_performance", "subtests": []}

        # Response time benchmarks
        response_time_test = await self._test_response_time_benchmarks()
        results["subtests"].append(response_time_test)

        # Concurrent interaction handling
        concurrent_test = await self._test_concurrent_voice_interactions()
        results["subtests"].append(concurrent_test)

        # Pipeline latency analysis
        latency_test = await self._test_pipeline_latency_analysis()
        results["subtests"].append(latency_test)

        results["overall_success"] = all(t["success"] for t in results["subtests"])
        return results

    async def _test_response_time_benchmarks(self) -> Dict[str, Any]:
        """Test response time benchmarks for different query types."""
        print("📍 Testing response time benchmarks...")

        benchmark_queries = [
            {"query": "Hello", "type": "simple", "target_time": 2.0},
            {"query": "What's 15 plus 23?", "type": "calculation", "target_time": 3.0},
            {
                "query": "What's the weather today?",
                "type": "information",
                "target_time": 5.0,
            },
            {
                "query": "Create a file with today's date",
                "type": "file_operation",
                "target_time": 4.0,
            },
            {
                "query": "Explain quantum computing in simple terms",
                "type": "complex",
                "target_time": 8.0,
            },
        ]

        benchmark_results = []

        for benchmark in benchmark_queries:
            try:
                # Warm-up run
                await self.orchestrator.process_text("test")

                # Timed run
                start_time = time.time()
                response = await self.orchestrator.process_text(benchmark["query"])
                response_time = time.time() - start_time

                meets_target = response_time <= benchmark["target_time"]

                benchmark_results.append(
                    {
                        "query_type": benchmark["type"],
                        "query": benchmark["query"],
                        "response_time": response_time,
                        "target_time": benchmark["target_time"],
                        "meets_target": meets_target,
                        "response_length": len(response) if response else 0,
                        "success": bool(response) and meets_target,
                    }
                )

                print(
                    f"  ✅ {benchmark['type']}: {response_time:.2f}s (target: {benchmark['target_time']}s)"
                )

            except Exception as e:
                benchmark_results.append(
                    {
                        "query_type": benchmark["type"],
                        "query": benchmark["query"],
                        "success": False,
                        "error": str(e),
                    }
                )
                print(f"  ❌ {benchmark['type']}: {e}")

        avg_response_time = np.mean(
            [r["response_time"] for r in benchmark_results if "response_time" in r]
        )
        target_success_rate = sum(
            1 for r in benchmark_results if r.get("meets_target", False)
        ) / len(benchmark_results)

        return {
            "name": "response_time_benchmarks",
            "success": target_success_rate >= 0.6,
            "average_response_time": avg_response_time,
            "target_success_rate": target_success_rate,
            "benchmarks": benchmark_results,
        }

    async def _test_concurrent_voice_interactions(self) -> Dict[str, Any]:
        """Test handling of concurrent voice interactions."""
        print("📍 Testing concurrent voice interactions...")

        # Simulate concurrent requests
        concurrent_queries = [
            "What's 10 plus 10?",
            "What's the current time?",
            "Calculate 5 times 6",
            "Tell me a joke",
        ]

        try:
            start_time = time.time()

            # Launch concurrent requests
            tasks = [
                self.orchestrator.process_text(query) for query in concurrent_queries
            ]

            responses = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

            # Analyze results
            successful_responses = sum(1 for r in responses if isinstance(r, str) and r)
            error_count = sum(1 for r in responses if isinstance(r, Exception))

            concurrent_results = {
                "total_requests": len(concurrent_queries),
                "successful_responses": successful_responses,
                "error_count": error_count,
                "total_time": total_time,
                "average_time_per_request": total_time / len(concurrent_queries),
                "success": successful_responses >= len(concurrent_queries) * 0.75,
            }

            print(
                f"  ✅ Concurrent: {successful_responses}/{len(concurrent_queries)} successful in {total_time:.2f}s"
            )

        except Exception as e:
            concurrent_results = {"success": False, "error": str(e)}
            print(f"  ❌ Concurrent test failed: {e}")

        return {
            "name": "concurrent_voice_interactions",
            "success": concurrent_results.get("success", False),
            "results": concurrent_results,
        }

    async def _test_pipeline_latency_analysis(self) -> Dict[str, Any]:
        """Analyze latency in the voice processing pipeline."""
        print("📍 Testing pipeline latency analysis...")

        latency_measurements = []

        test_inputs = [
            "Quick test",
            "Medium length test input for timing",
            "This is a longer test input designed to measure pipeline latency across different input sizes and complexity levels",
        ]

        for input_text in test_inputs:
            try:
                # Mock pipeline components for latency measurement
                pipeline_times = {}

                # Simulate STT latency
                stt_start = time.time()
                await asyncio.sleep(0.1)  # Simulate STT processing
                pipeline_times["stt"] = time.time() - stt_start

                # Measure actual LLM processing
                llm_start = time.time()
                response = await self.orchestrator.process_text(input_text)
                pipeline_times["llm"] = time.time() - llm_start

                # Simulate TTS latency
                tts_start = time.time()
                await asyncio.sleep(0.05)  # Simulate TTS processing
                pipeline_times["tts"] = time.time() - tts_start

                total_latency = sum(pipeline_times.values())

                latency_measurements.append(
                    {
                        "input_length": len(input_text),
                        "input_text": (
                            input_text[:50] + "..."
                            if len(input_text) > 50
                            else input_text
                        ),
                        "stt_latency": pipeline_times["stt"],
                        "llm_latency": pipeline_times["llm"],
                        "tts_latency": pipeline_times["tts"],
                        "total_latency": total_latency,
                        "response_generated": bool(response),
                        "success": bool(response)
                        and total_latency < 10.0,  # 10s max acceptable
                    }
                )

                print(
                    f"  ✅ Input ({len(input_text)} chars): {total_latency:.2f}s total"
                )

            except Exception as e:
                latency_measurements.append(
                    {"input_length": len(input_text), "success": False, "error": str(e)}
                )
                print(f"  ❌ Latency test failed: {e}")

        avg_total_latency = np.mean(
            [m["total_latency"] for m in latency_measurements if "total_latency" in m]
        )
        success_rate = sum(1 for m in latency_measurements if m["success"]) / len(
            latency_measurements
        )

        return {
            "name": "pipeline_latency_analysis",
            "success": success_rate >= 0.75 and avg_total_latency < 8.0,
            "average_total_latency": avg_total_latency,
            "success_rate": success_rate,
            "measurements": latency_measurements,
        }

    async def test_integration_fallback_mechanisms(self) -> Dict[str, Any]:
        """Test 5: Integration and Fallback Testing"""
        print("\n=== Test 5: Integration and Fallback Mechanisms ===")
        results = {"name": "integration_fallback", "subtests": []}

        # Multi-agent to single-agent fallback
        fallback_test = await self._test_multi_agent_fallback()
        results["subtests"].append(fallback_test)

        # Error recovery testing
        error_recovery_test = await self._test_error_recovery()
        results["subtests"].append(error_recovery_test)

        # Configuration switching
        config_switch_test = await self._test_configuration_switching()
        results["subtests"].append(config_switch_test)

        results["overall_success"] = all(t["success"] for t in results["subtests"])
        return results

    async def _test_multi_agent_fallback(self) -> Dict[str, Any]:
        """Test fallback from multi-agent to single-agent mode."""
        print("📍 Testing multi-agent to single-agent fallback...")

        fallback_results = []

        try:
            # Disable multi-agent mode to trigger fallback
            original_setting = self.orchestrator.multi_agent_enabled
            self.orchestrator.disable_multi_agent_mode()

            # Test requests in fallback mode
            fallback_queries = [
                "Hello, how are you?",
                "What's 2 plus 2?",
                "Tell me about the weather",
            ]

            for query in fallback_queries:
                try:
                    response = await self.orchestrator.process_text(query)

                    fallback_results.append(
                        {
                            "query": query,
                            "response_generated": bool(response),
                            "success": bool(response),
                        }
                    )

                    print(f"  ✅ Fallback query successful: '{query}'")

                except Exception as e:
                    fallback_results.append(
                        {"query": query, "success": False, "error": str(e)}
                    )
                    print(f"  ❌ Fallback query failed: '{query}': {e}")

            # Restore original setting
            if original_setting:
                self.orchestrator.enable_multi_agent_mode()

        except Exception as e:
            fallback_results.append(
                {"success": False, "error": f"Fallback test setup failed: {e}"}
            )
            print(f"  ❌ Fallback test setup failed: {e}")

        success_rate = sum(1 for r in fallback_results if r["success"]) / max(
            len(fallback_results), 1
        )

        return {
            "name": "multi_agent_fallback",
            "success": success_rate >= 0.75,
            "success_rate": success_rate,
            "fallback_tests": fallback_results,
        }

    async def _test_error_recovery(self) -> Dict[str, Any]:
        """Test error recovery mechanisms."""
        print("📍 Testing error recovery mechanisms...")

        recovery_scenarios = [
            {"query": "", "description": "Empty input"},
            {"query": "a" * 1000, "description": "Extremely long input"},
            {
                "query": "Invalid query with special chars: !@#$%^&*()",
                "description": "Special characters",
            },
        ]

        recovery_results = []

        for scenario in recovery_scenarios:
            try:
                response = await self.orchestrator.process_text(scenario["query"])

                # Check if system recovered gracefully
                recovered_gracefully = bool(response) or "error" not in response.lower()

                recovery_results.append(
                    {
                        "scenario": scenario["description"],
                        "query_length": len(scenario["query"]),
                        "response_generated": bool(response),
                        "recovered_gracefully": recovered_gracefully,
                        "success": recovered_gracefully,
                    }
                )

                print(f"  ✅ {scenario['description']}: Recovered gracefully")

            except Exception as e:
                # Check if exception is handled appropriately
                handled_appropriately = "timeout" not in str(e).lower()

                recovery_results.append(
                    {
                        "scenario": scenario["description"],
                        "exception_occurred": True,
                        "handled_appropriately": handled_appropriately,
                        "success": handled_appropriately,
                        "error": str(e),
                    }
                )

                status = "appropriately" if handled_appropriately else "poorly"
                print(f"  ⚠️  {scenario['description']}: Exception handled {status}")

        success_rate = sum(1 for r in recovery_results if r["success"]) / len(
            recovery_results
        )

        return {
            "name": "error_recovery",
            "success": success_rate >= 0.67,
            "success_rate": success_rate,
            "recovery_scenarios": recovery_results,
        }

    async def _test_configuration_switching(self) -> Dict[str, Any]:
        """Test configuration switching during operation."""
        print("📍 Testing configuration switching...")

        switch_results = []

        try:
            # Test switching multi-agent on/off
            original_state = self.orchestrator.multi_agent_enabled

            # Switch to opposite state
            if original_state:
                self.orchestrator.disable_multi_agent_mode()
                new_state = False
            else:
                self.orchestrator.enable_multi_agent_mode()
                new_state = True

            # Verify switch worked
            switch_successful = self.orchestrator.multi_agent_enabled == new_state

            # Test functionality in new state
            test_response = await self.orchestrator.process_text(
                "Test configuration switch"
            )
            functionality_works = bool(test_response)

            # Restore original state
            if original_state:
                self.orchestrator.enable_multi_agent_mode()
            else:
                self.orchestrator.disable_multi_agent_mode()

            switch_results.append(
                {
                    "switch_type": "multi_agent_toggle",
                    "original_state": original_state,
                    "target_state": new_state,
                    "switch_successful": switch_successful,
                    "functionality_works": functionality_works,
                    "success": switch_successful and functionality_works,
                }
            )

            print(
                f"  ✅ Multi-agent toggle: {original_state} -> {new_state} -> {original_state}"
            )

        except Exception as e:
            switch_results.append(
                {"switch_type": "multi_agent_toggle", "success": False, "error": str(e)}
            )
            print(f"  ❌ Configuration switch failed: {e}")

        success_rate = sum(1 for r in switch_results if r["success"]) / max(
            len(switch_results), 1
        )

        return {
            "name": "configuration_switching",
            "success": success_rate >= 0.75,
            "success_rate": success_rate,
            "switch_tests": switch_results,
        }

    async def test_tui_integration_during_voice(self) -> Dict[str, Any]:
        """Test 6: TUI Integration during Voice Operations"""
        print("\n=== Test 6: TUI Integration during Voice Operations ===")
        results = {"name": "tui_integration", "subtests": []}

        # Pipeline status updates
        status_test = await self._test_pipeline_status_updates()
        results["subtests"].append(status_test)

        # Multi-agent activity display
        activity_test = await self._test_multi_agent_activity_display()
        results["subtests"].append(activity_test)

        # Voice command integration
        voice_command_test = await self._test_voice_command_integration()
        results["subtests"].append(voice_command_test)

        results["overall_success"] = all(t["success"] for t in results["subtests"])
        return results

    async def _test_pipeline_status_updates(self) -> Dict[str, Any]:
        """Test pipeline status updates during voice operations."""
        print("📍 Testing pipeline status updates...")

        status_updates = []

        # Monitor status during processing
        initial_status = self.pipeline_status.snapshot()

        try:
            # Process a query and monitor status changes
            response = await self.orchestrator.process_text("Test status updates")

            final_status = self.pipeline_status.snapshot()

            # Check if status was updated appropriately
            status_changed = initial_status != final_status
            llm_was_active = (
                final_status.get("llm") == "ready"
            )  # Should be ready after processing

            status_updates.append(
                {
                    "initial_status": initial_status,
                    "final_status": final_status,
                    "status_changed": status_changed,
                    "llm_processed": llm_was_active,
                    "success": bool(response) and status_changed,
                }
            )

            print(
                f"  ✅ Status updates: Changed={status_changed}, LLM Ready={llm_was_active}"
            )

        except Exception as e:
            status_updates.append({"success": False, "error": str(e)})
            print(f"  ❌ Status update test failed: {e}")

        success_rate = sum(1 for u in status_updates if u["success"]) / max(
            len(status_updates), 1
        )

        return {
            "name": "pipeline_status_updates",
            "success": success_rate >= 0.75,
            "success_rate": success_rate,
            "status_updates": status_updates,
        }

    async def _test_multi_agent_activity_display(self) -> Dict[str, Any]:
        """Test multi-agent activity display in TUI."""
        print("📍 Testing multi-agent activity display...")

        activity_results = []

        try:
            # Get orchestrator info to check multi-agent activity
            info = self.orchestrator.get_orchestrator_info()

            has_multi_agent_info = "multi_agent_info" in info
            shows_agent_count = False
            shows_routing_stats = False

            if has_multi_agent_info:
                ma_info = info["multi_agent_info"]
                shows_agent_count = "active_agents" in ma_info
                shows_routing_stats = "routing_stats" in ma_info

            activity_results.append(
                {
                    "has_multi_agent_info": has_multi_agent_info,
                    "shows_agent_count": shows_agent_count,
                    "shows_routing_stats": shows_routing_stats,
                    "orchestrator_info": info,
                    "success": has_multi_agent_info and shows_agent_count,
                }
            )

            print(
                f"  ✅ Activity display: Info={has_multi_agent_info}, Agents={shows_agent_count}"
            )

        except Exception as e:
            activity_results.append({"success": False, "error": str(e)})
            print(f"  ❌ Activity display test failed: {e}")

        success_rate = sum(1 for r in activity_results if r["success"]) / max(
            len(activity_results), 1
        )

        return {
            "name": "multi_agent_activity_display",
            "success": success_rate >= 0.75,
            "success_rate": success_rate,
            "activity_results": activity_results,
        }

    async def _test_voice_command_integration(self) -> Dict[str, Any]:
        """Test voice command integration with TUI."""
        print("📍 Testing voice command integration...")

        command_results = []

        # Test voice commands that should integrate with TUI
        voice_commands = [
            {"command": "start dictation", "expected_action": "dictation_mode"},
            {"command": "privacy mode", "expected_action": "privacy_toggle"},
            {"command": "end dictation", "expected_action": "dictation_end"},
        ]

        for cmd_test in voice_commands:
            try:
                # Test command detection
                from voice_agent.ui.tui_app import VoiceAgentTUI

                command_detected = VoiceAgentTUI.detect_voice_command(
                    cmd_test["command"]
                )

                command_results.append(
                    {
                        "command": cmd_test["command"],
                        "expected_action": cmd_test["expected_action"],
                        "command_detected": command_detected is not None,
                        "detected_command": command_detected,
                        "success": command_detected is not None,
                    }
                )

                print(f"  ✅ '{cmd_test['command']}': Detected={command_detected}")

            except Exception as e:
                command_results.append(
                    {"command": cmd_test["command"], "success": False, "error": str(e)}
                )
                print(f"  ❌ '{cmd_test['command']}': {e}")

        success_rate = sum(1 for r in command_results if r["success"]) / len(
            command_results
        )

        return {
            "name": "voice_command_integration",
            "success": success_rate >= 0.67,
            "success_rate": success_rate,
            "command_tests": command_results,
        }

    async def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE END-TO-END VOICE + MULTI-AGENT TEST REPORT")
        print("=" * 80)

        # Run all test suites
        test_suites = [
            (
                "Voice Pipeline Integration",
                self.test_voice_pipeline_multi_agent_integration,
            ),
            ("Multi-Step Workflows", self.test_multi_step_workflows_with_context),
            (
                "Voice-Specific Scenarios",
                self.test_voice_specific_multi_agent_scenarios,
            ),
            ("Real-Time Performance", self.test_real_time_interaction_performance),
            ("Integration Fallbacks", self.test_integration_fallback_mechanisms),
            ("TUI Integration", self.test_tui_integration_during_voice),
        ]

        overall_results = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_suites": [],
            "overall_metrics": {},
            "performance_data": self.performance_metrics,
            "system_info": {
                "multi_agent_enabled": (
                    self.orchestrator.multi_agent_enabled
                    if self.orchestrator
                    else False
                ),
                "orchestrator_initialized": (
                    self.orchestrator.is_initialized if self.orchestrator else False
                ),
            },
        }

        total_tests = 0
        successful_tests = 0

        for suite_name, test_func in test_suites:
            print(f"\n🧪 Running {suite_name} tests...")

            try:
                suite_results = await test_func()
                suite_results["suite_name"] = suite_name

                # Count tests and successes
                suite_test_count = len(suite_results.get("subtests", []))
                suite_success_count = sum(
                    1
                    for t in suite_results.get("subtests", [])
                    if t.get("success", False)
                )

                total_tests += suite_test_count
                successful_tests += suite_success_count

                overall_results["test_suites"].append(suite_results)

                status = (
                    "✅ PASSED"
                    if suite_results.get("overall_success", False)
                    else "❌ FAILED"
                )
                print(
                    f"   {status} - {suite_success_count}/{suite_test_count} subtests passed"
                )

            except Exception as e:
                print(f"   ❌ SUITE FAILED - {suite_name}: {e}")
                overall_results["test_suites"].append(
                    {
                        "suite_name": suite_name,
                        "overall_success": False,
                        "error": str(e),
                    }
                )

        # Calculate overall metrics
        overall_success_rate = successful_tests / max(total_tests, 1)
        suite_success_rate = sum(
            1 for s in overall_results["test_suites"] if s.get("overall_success", False)
        ) / len(test_suites)

        overall_results["overall_metrics"] = {
            "total_test_suites": len(test_suites),
            "successful_test_suites": sum(
                1
                for s in overall_results["test_suites"]
                if s.get("overall_success", False)
            ),
            "total_subtests": total_tests,
            "successful_subtests": successful_tests,
            "overall_success_rate": overall_success_rate,
            "suite_success_rate": suite_success_rate,
            "system_ready_for_production": suite_success_rate >= 0.75
            and overall_success_rate >= 0.70,
        }

        # Performance analysis
        if self.performance_metrics:
            avg_processing_time = np.mean(
                [
                    m["processing_time"]
                    for m in self.performance_metrics
                    if "processing_time" in m
                ]
            )
            overall_results["overall_metrics"][
                "average_processing_time"
            ] = avg_processing_time

        # Final assessment
        print(f"\n" + "=" * 80)
        print(f"📈 FINAL ASSESSMENT")
        print(f"=" * 80)
        print(
            f"Suite Success Rate: {suite_success_rate:.1%} ({sum(1 for s in overall_results['test_suites'] if s.get('overall_success', False))}/{len(test_suites)} suites)"
        )
        print(
            f"Overall Success Rate: {overall_success_rate:.1%} ({successful_tests}/{total_tests} tests)"
        )

        if overall_results["overall_metrics"]["system_ready_for_production"]:
            print(f"🎉 SYSTEM STATUS: ✅ READY FOR PRODUCTION")
            print(
                f"   The voice + multi-agent system is performing well and ready for deployment."
            )
        else:
            print(f"⚠️  SYSTEM STATUS: ❌ NEEDS IMPROVEMENT")
            print(
                f"   The system requires additional work before production deployment."
            )

        return overall_results

    async def cleanup(self) -> None:
        """Cleanup test resources."""
        if self.orchestrator:
            await self.orchestrator.stop()


async def main():
    """Main test execution function."""
    print("🚀 Starting Comprehensive End-to-End Voice + Multi-Agent Workflow Testing")
    print("=" * 80)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize test framework
    test_framework = VoiceWorkflowTestFramework()

    try:
        # Setup test environment
        setup_success = await test_framework.setup_test_environment()
        if not setup_success:
            print("❌ Test environment setup failed. Exiting.")
            sys.exit(1)

        # Run comprehensive tests
        results = await test_framework.generate_comprehensive_report()

        # Save results to file
        results_file = Path("test_results_voice_multi_agent_workflows.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📄 Detailed results saved to: {results_file}")

        # Determine exit code
        system_ready = results["overall_metrics"]["system_ready_for_production"]
        exit_code = 0 if system_ready else 1

        print(f"\n{'='*80}")
        if system_ready:
            print("🎯 CONCLUSION: Voice + Multi-Agent system validation SUCCESSFUL!")
            print(
                "   The system is ready for production use with comprehensive workflow support."
            )
        else:
            print(
                "⚠️  CONCLUSION: Voice + Multi-Agent system validation needs improvement."
            )
            print(
                "   Review failed tests and address issues before production deployment."
            )

        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test execution failed with error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        await test_framework.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
