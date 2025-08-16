#!/usr/bin/env python3
"""
Multi-Agent Voice System Examples

This script demonstrates various multi-agent capabilities including:
- Basic agent routing
- Multi-step workflows
- Agent collaboration
- Complex task delegation
- Voice interaction patterns

Usage:
    python examples/multi_agent_examples.py [example_name]

Available examples:
    - basic_routing: Test basic agent routing
    - workflows: Multi-step workflow examples
    - collaboration: Agent collaboration examples
    - voice_patterns: Voice interaction examples
    - all: Run all examples
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_agent.core.config import Config
from voice_agent.core.voice_agent_orchestrator import VoiceAgentOrchestrator


class MultiAgentExamples:
    """Comprehensive multi-agent examples and demonstrations."""

    def __init__(self):
        """Initialize the examples system."""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.config = None
        self.orchestrator = None

    def setup_logging(self):
        """Setup logging for examples."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    async def initialize(self):
        """Initialize the multi-agent system."""
        try:
            # Load configuration
            config_path = (
                Path(__file__).parent.parent
                / "src"
                / "voice_agent"
                / "config"
                / "default.yaml"
            )
            self.config = Config.load(config_path)

            # Enable multi-agent features
            self.config.multi_agent.enabled = True

            # Create orchestrator
            self.orchestrator = VoiceAgentOrchestrator(config=self.config)
            await self.orchestrator.initialize()

            self.logger.info("Multi-agent system initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize multi-agent system: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        if self.orchestrator:
            await self.orchestrator.cleanup()

    async def basic_routing_examples(self):
        """Demonstrate basic agent routing capabilities."""
        self.logger.info("=== Basic Agent Routing Examples ===")

        test_queries = [
            {
                "query": "What is 25 * 7 + 12?",
                "expected_agent": "utility_agent",
                "description": "Mathematical calculation should route to UtilityAgent",
            },
            {
                "query": "What's the weather like in London?",
                "expected_agent": "information_agent",
                "description": "Weather query should route to InformationAgent",
            },
            {
                "query": "List files in the current directory",
                "expected_agent": "productivity_agent",
                "description": "File operation should route to ProductivityAgent",
            },
            {
                "query": "Hello, how are you today?",
                "expected_agent": "general_agent",
                "description": "General conversation should route to GeneralAgent",
            },
            {
                "query": "Search for Python programming tutorials",
                "expected_agent": "information_agent",
                "description": "Web search should route to InformationAgent",
            },
        ]

        for test in test_queries:
            self.logger.info(f"\n--- Testing: {test['description']} ---")
            self.logger.info(f"Query: '{test['query']}'")

            try:
                # Process the query
                response = await self.orchestrator.process_text(test["query"])

                self.logger.info(f"Response: {response}")
                self.logger.info("✅ Query processed successfully")

            except Exception as e:
                self.logger.error(f"❌ Error processing query: {e}")

            # Small delay between queries
            await asyncio.sleep(1)

    async def workflow_examples(self):
        """Demonstrate multi-step workflow capabilities."""
        self.logger.info("\n=== Multi-Step Workflow Examples ===")

        workflows = [
            {
                "name": "Weather and File Workflow",
                "query": "Get weather for Tokyo and save it to weather_tokyo.txt",
                "description": "Sequential workflow: weather retrieval → file saving",
                "expected_flow": [
                    "InformationAgent gets weather data",
                    "ProductivityAgent saves to file",
                    "Results aggregated and returned",
                ],
            },
            {
                "name": "Research and Analysis Workflow",
                "query": "Search for renewable energy trends and calculate growth percentage",
                "description": "Pipeline workflow: research → analysis → calculation",
                "expected_flow": [
                    "InformationAgent searches for information",
                    "UtilityAgent performs calculations",
                    "Results combined",
                ],
            },
            {
                "name": "Financial Analysis Workflow",
                "query": "Calculate compound interest for $10000 at 5% for 10 years and save the analysis",
                "description": "Multi-step: calculation → explanation → file save",
                "expected_flow": [
                    "UtilityAgent calculates compound interest",
                    "GeneralAgent provides explanation",
                    "ProductivityAgent saves analysis",
                ],
            },
        ]

        for workflow in workflows:
            self.logger.info(f"\n--- {workflow['name']} ---")
            self.logger.info(f"Description: {workflow['description']}")
            self.logger.info(f"Query: '{workflow['query']}'")
            self.logger.info("Expected flow:")
            for step in workflow["expected_flow"]:
                self.logger.info(f"  → {step}")

            try:
                # Process the workflow
                response = await self.orchestrator.process_text(workflow["query"])

                self.logger.info(f"Result: {response}")
                self.logger.info("✅ Workflow completed successfully")

            except Exception as e:
                self.logger.error(f"❌ Workflow failed: {e}")

            await asyncio.sleep(2)

    async def collaboration_examples(self):
        """Demonstrate agent collaboration capabilities."""
        self.logger.info("\n=== Agent Collaboration Examples ===")

        collaboration_tasks = [
            {
                "name": "Multi-Perspective Analysis",
                "query": "Analyze the benefits of renewable energy from both economic and environmental perspectives",
                "description": "Multiple agents provide different analytical perspectives",
                "agents_involved": [
                    "information_agent",
                    "utility_agent",
                    "general_agent",
                ],
            },
            {
                "name": "Consensus Calculation",
                "query": "Verify this calculation using multiple methods: What is the square root of 144?",
                "description": "Multiple agents confirm calculation accuracy",
                "agents_involved": ["utility_agent", "general_agent"],
            },
            {
                "name": "Comprehensive Research",
                "query": "Research artificial intelligence trends and create a comprehensive report with data analysis",
                "description": "Coordinated research, analysis, and report generation",
                "agents_involved": [
                    "information_agent",
                    "utility_agent",
                    "productivity_agent",
                ],
            },
        ]

        for task in collaboration_tasks:
            self.logger.info(f"\n--- {task['name']} ---")
            self.logger.info(f"Description: {task['description']}")
            self.logger.info(f"Agents involved: {', '.join(task['agents_involved'])}")
            self.logger.info(f"Query: '{task['query']}'")

            try:
                # Process collaborative task
                response = await self.orchestrator.process_text(task["query"])

                self.logger.info(f"Collaborative result: {response}")
                self.logger.info("✅ Collaboration completed successfully")

            except Exception as e:
                self.logger.error(f"❌ Collaboration failed: {e}")

            await asyncio.sleep(3)

    async def voice_interaction_patterns(self):
        """Demonstrate voice interaction patterns and responses."""
        self.logger.info("\n=== Voice Interaction Patterns ===")

        voice_examples = [
            {
                "category": "Greeting and Setup",
                "examples": [
                    "Hello, I need help with some calculations",
                    "Hi there, can you help me with weather information?",
                    "Good morning, I'd like to organize some files",
                ],
            },
            {
                "category": "Task Requests",
                "examples": [
                    "Please calculate the monthly payment for a $200,000 mortgage at 4.5% for 30 years",
                    "Can you get the current weather for San Francisco and save it to a file?",
                    "I need you to search for Python tutorials and summarize the top results",
                ],
            },
            {
                "category": "Follow-up Questions",
                "examples": [
                    "Can you explain that calculation in more detail?",
                    "What about the weather for tomorrow?",
                    "Save that information to a file called results.txt",
                ],
            },
            {
                "category": "Complex Workflows",
                "examples": [
                    "Get weather for London, calculate heating costs if temperature is below 10°C, and create a report",
                    "Search for stock market trends, analyze the data, and save a summary report",
                    "Calculate my monthly expenses, compare with last month, and file the analysis",
                ],
            },
        ]

        for category in voice_examples:
            self.logger.info(f"\n--- {category['category']} ---")

            for example in category["examples"]:
                self.logger.info(f"\nExample: '{example}'")

                try:
                    # Simulate voice processing
                    response = await self.orchestrator.process_text(example)

                    self.logger.info(f"Response: {response}")
                    self.logger.info("✅ Voice pattern processed")

                except Exception as e:
                    self.logger.error(f"❌ Voice pattern failed: {e}")

                await asyncio.sleep(1)

    async def performance_examples(self):
        """Demonstrate performance and optimization features."""
        self.logger.info("\n=== Performance and Optimization Examples ===")

        performance_tests = [
            {
                "name": "Parallel Processing",
                "query": "Get weather for New York, London, and Tokyo simultaneously",
                "description": "Multiple InformationAgents work in parallel",
            },
            {
                "name": "Load Balancing",
                "queries": [
                    "Calculate 15% of 250",
                    "What is 75 divided by 3?",
                    "Find the square root of 169",
                    "Calculate 20% tip on $45.50",
                ],
                "description": "Multiple calculation requests distributed across agents",
            },
            {
                "name": "Context Preservation",
                "queries": [
                    "What's the weather in Paris?",
                    "Save that weather information to paris_weather.txt",
                    "Now get the weather for London",
                    "Compare the two weather reports",
                ],
                "description": "Context maintained across multiple agent handoffs",
            },
        ]

        for test in performance_tests:
            self.logger.info(f"\n--- {test['name']} ---")
            self.logger.info(f"Description: {test['description']}")

            if "query" in test:
                # Single query test
                self.logger.info(f"Query: '{test['query']}'")

                try:
                    start_time = asyncio.get_event_loop().time()
                    response = await self.orchestrator.process_text(test["query"])
                    end_time = asyncio.get_event_loop().time()

                    self.logger.info(f"Response: {response}")
                    self.logger.info(
                        f"⏱️ Processing time: {end_time - start_time:.2f} seconds"
                    )
                    self.logger.info("✅ Performance test completed")

                except Exception as e:
                    self.logger.error(f"❌ Performance test failed: {e}")

            elif "queries" in test:
                # Multiple query test
                for i, query in enumerate(test["queries"], 1):
                    self.logger.info(f"Query {i}: '{query}'")

                    try:
                        start_time = asyncio.get_event_loop().time()
                        response = await self.orchestrator.process_text(query)
                        end_time = asyncio.get_event_loop().time()

                        self.logger.info(f"Response: {response}")
                        self.logger.info(f"⏱️ Time: {end_time - start_time:.2f}s")

                    except Exception as e:
                        self.logger.error(f"❌ Query {i} failed: {e}")

                    await asyncio.sleep(0.5)

            await asyncio.sleep(2)

    async def run_example(self, example_name: str):
        """Run a specific example."""
        examples = {
            "basic_routing": self.basic_routing_examples,
            "workflows": self.workflow_examples,
            "collaboration": self.collaboration_examples,
            "voice_patterns": self.voice_interaction_patterns,
            "performance": self.performance_examples,
        }

        if example_name not in examples:
            self.logger.error(f"Unknown example: {example_name}")
            self.logger.info(f"Available examples: {', '.join(examples.keys())}")
            return

        await examples[example_name]()

    async def run_all_examples(self):
        """Run all available examples."""
        self.logger.info("🚀 Running All Multi-Agent Examples")

        examples = [
            ("basic_routing", self.basic_routing_examples),
            ("workflows", self.workflow_examples),
            ("collaboration", self.collaboration_examples),
            ("voice_patterns", self.voice_interaction_patterns),
            ("performance", self.performance_examples),
        ]

        for name, example_func in examples:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Running Example: {name.replace('_', ' ').title()}")
            self.logger.info(f"{'='*60}")

            try:
                await example_func()
                self.logger.info(f"✅ {name} examples completed successfully")

            except Exception as e:
                self.logger.error(f"❌ {name} examples failed: {e}")

            # Pause between example sets
            await asyncio.sleep(3)

        self.logger.info("\n🎉 All multi-agent examples completed!")


async def main():
    """Main function to run examples."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
    else:
        example_name = "all"

    # Create and run examples
    examples = MultiAgentExamples()

    try:
        await examples.initialize()

        if example_name == "all":
            await examples.run_all_examples()
        else:
            await examples.run_example(example_name)

    except KeyboardInterrupt:
        print("\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"❌ Examples failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await examples.cleanup()


if __name__ == "__main__":
    # Print usage information
    print("🤖 Multi-Agent Voice System Examples")
    print("=====================================")
    print()

    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print(__doc__)
        sys.exit(0)

    # Run examples
    asyncio.run(main())
