#!/usr/bin/env python3
"""
Individual Agent Demonstration Scripts

This script demonstrates each specialized agent's capabilities:
- InformationAgent: Research and information retrieval
- UtilityAgent: Mathematical calculations and analysis
- ProductivityAgent: File operations and task management
- GeneralAgent: Conversation and coordination
- ToolSpecialistAgent: Advanced tool operations

Usage:
    python examples/agent_demonstrations.py [agent_name]

Available agents:
    - information: InformationAgent demonstrations
    - utility: UtilityAgent demonstrations
    - productivity: ProductivityAgent demonstrations
    - general: GeneralAgent demonstrations
    - tool_specialist: ToolSpecialistAgent demonstrations
    - all: Run demonstrations for all agents
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_agent.core.config import Config
from voice_agent.core.voice_agent_orchestrator import VoiceAgentOrchestrator


class AgentDemonstrations:
    """Individual agent capability demonstrations."""

    def __init__(self):
        """Initialize the demonstration system."""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.config = None
        self.orchestrator = None

    def setup_logging(self):
        """Setup logging for demonstrations."""
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

            self.logger.info("Multi-agent system initialized for demonstrations")

        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        if self.orchestrator:
            await self.orchestrator.cleanup()

    async def demonstrate_information_agent(self):
        """Demonstrate InformationAgent capabilities."""
        self.logger.info("🔍 InformationAgent Demonstrations")
        self.logger.info("=" * 50)

        capabilities = [
            {
                "category": "Weather Information",
                "description": "Retrieving weather data for various locations",
                "examples": [
                    "What's the weather like in London?",
                    "Get the weather for New York in Fahrenheit",
                    "Check the weather in Tokyo and tell me if it's raining",
                    "Compare the weather between Paris and Berlin",
                ],
            },
            {
                "category": "Web Search",
                "description": "Searching the internet for information",
                "examples": [
                    "Search for Python programming tutorials",
                    "Find information about machine learning algorithms",
                    "Look up the latest news about renewable energy",
                    "Search for best practices in software development",
                ],
            },
            {
                "category": "News Retrieval",
                "description": "Getting current news and headlines",
                "examples": [
                    "Get the latest technology news",
                    "Find recent news about artificial intelligence",
                    "What are today's top headlines?",
                    "Search for news about climate change",
                ],
            },
            {
                "category": "Research Tasks",
                "description": "Complex information gathering and research",
                "examples": [
                    "Research the history of quantum computing",
                    "Find information about sustainable energy solutions",
                    "Look up statistics on global internet usage",
                    "Research the benefits of electric vehicles",
                ],
            },
        ]

        for capability in capabilities:
            self.logger.info(f"\n--- {capability['category']} ---")
            self.logger.info(f"Description: {capability['description']}")

            for example in capability["examples"]:
                self.logger.info(f"\nTesting: '{example}'")

                try:
                    response = await self.orchestrator.process_text(example)
                    self.logger.info(f"✅ Response: {response}")

                except Exception as e:
                    self.logger.error(f"❌ Error: {e}")

                await asyncio.sleep(1)

    async def demonstrate_utility_agent(self):
        """Demonstrate UtilityAgent capabilities."""
        self.logger.info("🧮 UtilityAgent Demonstrations")
        self.logger.info("=" * 50)

        capabilities = [
            {
                "category": "Basic Mathematics",
                "description": "Arithmetic operations and basic calculations",
                "examples": [
                    "What is 25 * 7 + 12?",
                    "Calculate 15% of 250",
                    "What's the square root of 144?",
                    "Find 20% tip on a $45.50 bill",
                ],
            },
            {
                "category": "Advanced Calculations",
                "description": "Complex mathematical operations",
                "examples": [
                    "Calculate compound interest for $10,000 at 5% annual rate for 10 years",
                    "What's the monthly payment for a $200,000 mortgage at 4.5% for 30 years?",
                    "Calculate the area of a circle with radius 7.5",
                    "Find the slope of a line passing through points (2,3) and (5,9)",
                ],
            },
            {
                "category": "Statistical Analysis",
                "description": "Data analysis and statistical calculations",
                "examples": [
                    "Calculate the average of these numbers: 12, 15, 18, 22, 33",
                    "Find the median of: 5, 8, 12, 15, 20, 25, 30",
                    "Calculate the percentage increase from 80 to 120",
                    "What's the standard deviation of: 2, 4, 6, 8, 10?",
                ],
            },
            {
                "category": "Unit Conversions",
                "description": "Converting between different units of measurement",
                "examples": [
                    "Convert 100 kilometers to miles",
                    "How many feet are in 5.5 meters?",
                    "Convert 75°F to Celsius",
                    "How many ounces are in 2.5 pounds?",
                ],
            },
        ]

        for capability in capabilities:
            self.logger.info(f"\n--- {capability['category']} ---")
            self.logger.info(f"Description: {capability['description']}")

            for example in capability["examples"]:
                self.logger.info(f"\nTesting: '{example}'")

                try:
                    response = await self.orchestrator.process_text(example)
                    self.logger.info(f"✅ Response: {response}")

                except Exception as e:
                    self.logger.error(f"❌ Error: {e}")

                await asyncio.sleep(1)

    async def demonstrate_productivity_agent(self):
        """Demonstrate ProductivityAgent capabilities."""
        self.logger.info("📁 ProductivityAgent Demonstrations")
        self.logger.info("=" * 50)

        capabilities = [
            {
                "category": "File Operations",
                "description": "File system operations and management",
                "examples": [
                    "List files in the current directory",
                    "Check if README.md exists",
                    "Create a new file called test_notes.txt",
                    "Save this data to a file: 'Hello, World!'",
                ],
            },
            {
                "category": "Directory Management",
                "description": "Working with directories and folder structures",
                "examples": [
                    "Show me the directory structure",
                    "List all Python files in the current directory",
                    "Check what files are in the examples folder",
                    "Find all configuration files",
                ],
            },
            {
                "category": "Data Organization",
                "description": "Organizing and structuring data",
                "examples": [
                    "Create a task list with priorities",
                    "Organize this information into categories",
                    "Create a simple project timeline",
                    "Make a checklist for project setup",
                ],
            },
            {
                "category": "Calendar and Scheduling",
                "description": "Time management and scheduling operations",
                "examples": [
                    "What's on my calendar today?",
                    "Schedule a meeting for tomorrow at 2 PM",
                    "Check my availability next week",
                    "Create a reminder for project deadline",
                ],
            },
        ]

        for capability in capabilities:
            self.logger.info(f"\n--- {capability['category']} ---")
            self.logger.info(f"Description: {capability['description']}")

            for example in capability["examples"]:
                self.logger.info(f"\nTesting: '{example}'")

                try:
                    response = await self.orchestrator.process_text(example)
                    self.logger.info(f"✅ Response: {response}")

                except Exception as e:
                    self.logger.error(f"❌ Error: {e}")

                await asyncio.sleep(1)

    async def demonstrate_general_agent(self):
        """Demonstrate GeneralAgent capabilities."""
        self.logger.info("💬 GeneralAgent Demonstrations")
        self.logger.info("=" * 50)

        capabilities = [
            {
                "category": "Conversational Interaction",
                "description": "Natural conversation and general assistance",
                "examples": [
                    "Hello, how are you today?",
                    "Can you help me understand how this system works?",
                    "What can you do for me?",
                    "Thank you for your help!",
                ],
            },
            {
                "category": "Explanations and Help",
                "description": "Providing explanations and assistance",
                "examples": [
                    "Explain how the multi-agent system works",
                    "What's the difference between the various agents?",
                    "How do I use voice commands effectively?",
                    "Can you give me tips for better interactions?",
                ],
            },
            {
                "category": "Task Coordination",
                "description": "Coordinating and routing complex requests",
                "examples": [
                    "I need help with both calculations and file operations",
                    "Can you coordinate a complex workflow for me?",
                    "Help me plan a multi-step project",
                    "Route this task to the appropriate specialist",
                ],
            },
            {
                "category": "General Knowledge",
                "description": "Providing general information and context",
                "examples": [
                    "Tell me about artificial intelligence",
                    "What are the benefits of voice interfaces?",
                    "Explain the concept of multi-agent systems",
                    "What's the history of voice recognition technology?",
                ],
            },
        ]

        for capability in capabilities:
            self.logger.info(f"\n--- {capability['category']} ---")
            self.logger.info(f"Description: {capability['description']}")

            for example in capability["examples"]:
                self.logger.info(f"\nTesting: '{example}'")

                try:
                    response = await self.orchestrator.process_text(example)
                    self.logger.info(f"✅ Response: {response}")

                except Exception as e:
                    self.logger.error(f"❌ Error: {e}")

                await asyncio.sleep(1)

    async def demonstrate_tool_specialist_agent(self):
        """Demonstrate ToolSpecialistAgent capabilities."""
        self.logger.info("🛠️ ToolSpecialistAgent Demonstrations")
        self.logger.info("=" * 50)

        capabilities = [
            {
                "category": "Advanced Tool Operations",
                "description": "Complex tool usage and chaining",
                "examples": [
                    "Use multiple tools to get weather and save it to a file",
                    "Chain calculator and file operations together",
                    "Perform a complex search and analysis workflow",
                    "Execute a multi-tool diagnostic sequence",
                ],
            },
            {
                "category": "System Operations",
                "description": "System-level operations and diagnostics",
                "examples": [
                    "Check system status and performance",
                    "Verify all tools are working correctly",
                    "Run a comprehensive system diagnostic",
                    "Test all available tool combinations",
                ],
            },
            {
                "category": "Tool Integration",
                "description": "Integrating and coordinating multiple tools",
                "examples": [
                    "Coordinate weather data retrieval with file storage",
                    "Integrate search results with calculation analysis",
                    "Combine calendar and file operations",
                    "Chain web search with data processing",
                ],
            },
            {
                "category": "Advanced Workflows",
                "description": "Complex multi-step tool workflows",
                "examples": [
                    "Execute a research-analyze-report workflow",
                    "Perform data collection and analysis pipeline",
                    "Run automated testing and validation sequence",
                    "Execute emergency diagnostic and response workflow",
                ],
            },
        ]

        for capability in capabilities:
            self.logger.info(f"\n--- {capability['category']} ---")
            self.logger.info(f"Description: {capability['description']}")

            for example in capability["examples"]:
                self.logger.info(f"\nTesting: '{example}'")

                try:
                    response = await self.orchestrator.process_text(example)
                    self.logger.info(f"✅ Response: {response}")

                except Exception as e:
                    self.logger.error(f"❌ Error: {e}")

                await asyncio.sleep(1)

    async def run_agent_demonstration(self, agent_name: str):
        """Run demonstration for a specific agent."""
        demonstrations = {
            "information": self.demonstrate_information_agent,
            "utility": self.demonstrate_utility_agent,
            "productivity": self.demonstrate_productivity_agent,
            "general": self.demonstrate_general_agent,
            "tool_specialist": self.demonstrate_tool_specialist_agent,
        }

        if agent_name not in demonstrations:
            self.logger.error(f"Unknown agent: {agent_name}")
            self.logger.info(f"Available agents: {', '.join(demonstrations.keys())}")
            return

        await demonstrations[agent_name]()

    async def run_all_demonstrations(self):
        """Run demonstrations for all agents."""
        self.logger.info("🤖 Running All Agent Demonstrations")
        self.logger.info("=" * 60)

        agents = [
            ("information", self.demonstrate_information_agent),
            ("utility", self.demonstrate_utility_agent),
            ("productivity", self.demonstrate_productivity_agent),
            ("general", self.demonstrate_general_agent),
            ("tool_specialist", self.demonstrate_tool_specialist_agent),
        ]

        for agent_name, demo_func in agents:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(
                f"Demonstrating: {agent_name.replace('_', ' ').title()}Agent"
            )
            self.logger.info(f"{'='*60}")

            try:
                await demo_func()
                self.logger.info(f"✅ {agent_name}Agent demonstrations completed")

            except Exception as e:
                self.logger.error(f"❌ {agent_name}Agent demonstrations failed: {e}")

            # Pause between agent demonstrations
            await asyncio.sleep(3)

        self.logger.info("\n🎉 All agent demonstrations completed!")


async def main():
    """Main function to run agent demonstrations."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        agent_name = sys.argv[1]
    else:
        agent_name = "all"

    # Create and run demonstrations
    demos = AgentDemonstrations()

    try:
        await demos.initialize()

        if agent_name == "all":
            await demos.run_all_demonstrations()
        else:
            await demos.run_agent_demonstration(agent_name)

    except KeyboardInterrupt:
        print("\n⏹️ Demonstrations interrupted by user")
    except Exception as e:
        print(f"❌ Demonstrations failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await demos.cleanup()


if __name__ == "__main__":
    # Print usage information
    print("🤖 Individual Agent Demonstrations")
    print("==================================")
    print()

    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print(__doc__)
        sys.exit(0)

    # Run demonstrations
    asyncio.run(main())
