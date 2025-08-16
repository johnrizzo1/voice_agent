#!/usr/bin/env python3
"""
Common Workflow Examples for Multi-Agent Voice System

This script demonstrates practical, real-world workflows that users commonly
perform with the multi-agent system. Each example shows how multiple agents
work together to accomplish complex tasks.

Usage:
    python examples/workflow_examples.py [workflow_name]

Available workflows:
    - daily_briefing: Morning briefing with weather, news, and schedule
    - research_report: Research a topic and create a comprehensive report
    - financial_analysis: Analyze financial data and generate insights
    - project_planning: Create project plans with tasks and timelines
    - data_processing: Process data through multiple analysis stages
    - travel_planning: Plan travel with weather, research, and organization
    - all: Run all workflow examples
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_agent.core.config import Config
from voice_agent.core.voice_agent_orchestrator import VoiceAgentOrchestrator


class WorkflowExamples:
    """Common multi-agent workflow demonstrations."""

    def __init__(self):
        """Initialize the workflow examples system."""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.config = None
        self.orchestrator = None

    def setup_logging(self):
        """Setup logging for workflow examples."""
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

            self.logger.info("Multi-agent system initialized for workflow examples")

        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        if self.orchestrator:
            await self.orchestrator.cleanup()

    async def daily_briefing_workflow(self):
        """Morning briefing workflow: weather, news, and schedule."""
        self.logger.info("🌅 Daily Briefing Workflow")
        self.logger.info("=" * 50)

        workflow_steps = [
            {
                "name": "Weather Check",
                "query": "What's the weather like today in New York?",
                "description": "Get current weather conditions",
            },
            {
                "name": "News Headlines",
                "query": "Get the latest technology news headlines",
                "description": "Fetch current news in areas of interest",
            },
            {
                "name": "Schedule Review",
                "query": "What's on my calendar for today?",
                "description": "Review today's appointments and tasks",
            },
            {
                "name": "Daily Summary",
                "query": "Create a daily briefing summary with weather, news, and schedule, then save it to daily_briefing.txt",
                "description": "Compile all information into a comprehensive briefing",
            },
        ]

        self.logger.info("Starting daily briefing workflow...")
        self.logger.info(
            "This workflow demonstrates how multiple agents collaborate for morning preparation"
        )

        briefing_data = {}

        for step in workflow_steps:
            self.logger.info(f"\n--- {step['name']} ---")
            self.logger.info(f"Description: {step['description']}")
            self.logger.info(f"Query: '{step['query']}'")

            try:
                response = await self.orchestrator.process_text(step["query"])
                briefing_data[step["name"]] = response
                self.logger.info(f"✅ Result: {response}")

            except Exception as e:
                self.logger.error(f"❌ Error in {step['name']}: {e}")
                briefing_data[step["name"]] = f"Error: {e}"

            await asyncio.sleep(2)

        # Display complete briefing
        self.logger.info("\n" + "=" * 60)
        self.logger.info("COMPLETE DAILY BRIEFING")
        self.logger.info("=" * 60)

        for step_name, result in briefing_data.items():
            self.logger.info(f"\n{step_name}:")
            self.logger.info(f"  {result}")

        self.logger.info("\n✅ Daily briefing workflow completed!")

    async def research_report_workflow(self):
        """Research and report generation workflow."""
        self.logger.info("📚 Research Report Workflow")
        self.logger.info("=" * 50)

        research_topic = "renewable energy trends"

        workflow_steps = [
            {
                "name": "Initial Research",
                "query": f"Search for information about {research_topic}",
                "description": "Gather initial information on the topic",
            },
            {
                "name": "Data Analysis",
                "query": "Based on the renewable energy research, calculate growth percentages and trends",
                "description": "Analyze numerical data and calculate key metrics",
            },
            {
                "name": "Market Research",
                "query": "Find recent news about renewable energy market developments",
                "description": "Get current market conditions and developments",
            },
            {
                "name": "Report Generation",
                "query": f"Create a comprehensive research report on {research_topic} including data analysis and market insights, then save it to research_report.md",
                "description": "Compile all research into a structured report",
            },
        ]

        self.logger.info(f"Researching topic: {research_topic}")
        self.logger.info("This workflow shows collaborative research and analysis")

        research_data = {}

        for step in workflow_steps:
            self.logger.info(f"\n--- {step['name']} ---")
            self.logger.info(f"Description: {step['description']}")
            self.logger.info(f"Query: '{step['query']}'")

            try:
                response = await self.orchestrator.process_text(step["query"])
                research_data[step["name"]] = response
                self.logger.info(f"✅ Result: {response}")

            except Exception as e:
                self.logger.error(f"❌ Error in {step['name']}: {e}")
                research_data[step["name"]] = f"Error: {e}"

            await asyncio.sleep(3)  # Longer delay for research tasks

        self.logger.info("\n✅ Research report workflow completed!")

    async def financial_analysis_workflow(self):
        """Financial analysis and planning workflow."""
        self.logger.info("💰 Financial Analysis Workflow")
        self.logger.info("=" * 50)

        workflow_steps = [
            {
                "name": "Investment Calculation",
                "query": "Calculate compound interest for $50,000 invested at 7% annual return for 20 years",
                "description": "Calculate long-term investment growth",
            },
            {
                "name": "Budget Analysis",
                "query": "Create a monthly budget breakdown for $5,000 income with 30% housing, 20% food, 15% transportation, 10% savings",
                "description": "Analyze budget allocation and remaining funds",
            },
            {
                "name": "Market Research",
                "query": "Search for current stock market trends and investment advice",
                "description": "Get current market conditions and recommendations",
            },
            {
                "name": "Financial Report",
                "query": "Create a comprehensive financial analysis report including investment projections, budget planning, and market insights, then save it to financial_analysis.txt",
                "description": "Compile financial analysis into actionable report",
            },
        ]

        self.logger.info("Performing comprehensive financial analysis...")
        self.logger.info("This workflow demonstrates financial planning and analysis")

        for step in workflow_steps:
            self.logger.info(f"\n--- {step['name']} ---")
            self.logger.info(f"Description: {step['description']}")
            self.logger.info(f"Query: '{step['query']}'")

            try:
                response = await self.orchestrator.process_text(step["query"])
                self.logger.info(f"✅ Result: {response}")

            except Exception as e:
                self.logger.error(f"❌ Error in {step['name']}: {e}")

            await asyncio.sleep(2)

        self.logger.info("\n✅ Financial analysis workflow completed!")

    async def project_planning_workflow(self):
        """Project planning and organization workflow."""
        self.logger.info("📋 Project Planning Workflow")
        self.logger.info("=" * 50)

        project_name = "Website Redesign Project"

        workflow_steps = [
            {
                "name": "Research Phase",
                "query": "Search for web design best practices and current trends",
                "description": "Research current best practices and trends",
            },
            {
                "name": "Timeline Calculation",
                "query": "Calculate project timeline: 2 weeks research, 4 weeks design, 3 weeks development, 1 week testing. What's the total project duration and end date if starting today?",
                "description": "Calculate project timeline and milestones",
            },
            {
                "name": "Task Organization",
                "query": f"Create a detailed task list for {project_name} with phases, dependencies, and priorities",
                "description": "Organize project tasks and dependencies",
            },
            {
                "name": "Project Documentation",
                "query": f"Create a comprehensive project plan for {project_name} including research findings, timeline, tasks, and save it to project_plan.md",
                "description": "Document complete project plan",
            },
        ]

        self.logger.info(f"Planning project: {project_name}")
        self.logger.info("This workflow shows project planning and organization")

        for step in workflow_steps:
            self.logger.info(f"\n--- {step['name']} ---")
            self.logger.info(f"Description: {step['description']}")
            self.logger.info(f"Query: '{step['query']}'")

            try:
                response = await self.orchestrator.process_text(step["query"])
                self.logger.info(f"✅ Result: {response}")

            except Exception as e:
                self.logger.error(f"❌ Error in {step['name']}: {e}")

            await asyncio.sleep(2)

        self.logger.info("\n✅ Project planning workflow completed!")

    async def data_processing_workflow(self):
        """Data processing and analysis workflow."""
        self.logger.info("📊 Data Processing Workflow")
        self.logger.info("=" * 50)

        workflow_steps = [
            {
                "name": "Data Collection",
                "query": "Get weather data for New York, London, Tokyo, Sydney, and Berlin",
                "description": "Collect data from multiple sources",
            },
            {
                "name": "Statistical Analysis",
                "query": "Calculate average temperature, identify highest and lowest temperatures, and find temperature ranges from the collected weather data",
                "description": "Perform statistical analysis on collected data",
            },
            {
                "name": "Data Visualization Planning",
                "query": "Create a plan for visualizing global weather data including chart types and key insights to highlight",
                "description": "Plan data presentation and visualization",
            },
            {
                "name": "Analysis Report",
                "query": "Create a comprehensive data analysis report including raw data, statistical analysis, and visualization recommendations, then save it to data_analysis_report.txt",
                "description": "Compile complete analysis report",
            },
        ]

        self.logger.info("Processing and analyzing multi-source data...")
        self.logger.info("This workflow demonstrates data processing pipeline")

        for step in workflow_steps:
            self.logger.info(f"\n--- {step['name']} ---")
            self.logger.info(f"Description: {step['description']}")
            self.logger.info(f"Query: '{step['query']}'")

            try:
                response = await self.orchestrator.process_text(step["query"])
                self.logger.info(f"✅ Result: {response}")

            except Exception as e:
                self.logger.error(f"❌ Error in {step['name']}: {e}")

            await asyncio.sleep(2)

        self.logger.info("\n✅ Data processing workflow completed!")

    async def travel_planning_workflow(self):
        """Travel planning and organization workflow."""
        self.logger.info("✈️ Travel Planning Workflow")
        self.logger.info("=" * 50)

        destination = "Tokyo, Japan"

        workflow_steps = [
            {
                "name": "Destination Research",
                "query": f"Search for travel information about {destination} including attractions, culture, and tips",
                "description": "Research destination information",
            },
            {
                "name": "Weather Planning",
                "query": f"What's the weather like in {destination} and what should I pack?",
                "description": "Check weather conditions for packing",
            },
            {
                "name": "Budget Calculation",
                "query": "Calculate travel budget: $1,200 flight, $150/night hotel for 7 nights, $80/day food for 8 days, $500 activities. What's the total cost and daily average?",
                "description": "Calculate travel costs and budget",
            },
            {
                "name": "Itinerary Creation",
                "query": f"Create a comprehensive travel itinerary for {destination} including research findings, weather considerations, budget breakdown, and save it to travel_plan.md",
                "description": "Compile complete travel plan",
            },
        ]

        self.logger.info(f"Planning travel to: {destination}")
        self.logger.info("This workflow shows comprehensive travel planning")

        for step in workflow_steps:
            self.logger.info(f"\n--- {step['name']} ---")
            self.logger.info(f"Description: {step['description']}")
            self.logger.info(f"Query: '{step['query']}'")

            try:
                response = await self.orchestrator.process_text(step["query"])
                self.logger.info(f"✅ Result: {response}")

            except Exception as e:
                self.logger.error(f"❌ Error in {step['name']}: {e}")

            await asyncio.sleep(2)

        self.logger.info("\n✅ Travel planning workflow completed!")

    async def run_workflow(self, workflow_name: str):
        """Run a specific workflow."""
        workflows = {
            "daily_briefing": self.daily_briefing_workflow,
            "research_report": self.research_report_workflow,
            "financial_analysis": self.financial_analysis_workflow,
            "project_planning": self.project_planning_workflow,
            "data_processing": self.data_processing_workflow,
            "travel_planning": self.travel_planning_workflow,
        }

        if workflow_name not in workflows:
            self.logger.error(f"Unknown workflow: {workflow_name}")
            self.logger.info(f"Available workflows: {', '.join(workflows.keys())}")
            return

        await workflows[workflow_name]()

    async def run_all_workflows(self):
        """Run all available workflows."""
        self.logger.info("🚀 Running All Workflow Examples")
        self.logger.info("=" * 60)

        workflows = [
            ("daily_briefing", self.daily_briefing_workflow),
            ("research_report", self.research_report_workflow),
            ("financial_analysis", self.financial_analysis_workflow),
            ("project_planning", self.project_planning_workflow),
            ("data_processing", self.data_processing_workflow),
            ("travel_planning", self.travel_planning_workflow),
        ]

        for workflow_name, workflow_func in workflows:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(
                f"Running Workflow: {workflow_name.replace('_', ' ').title()}"
            )
            self.logger.info(f"{'='*60}")

            try:
                await workflow_func()
                self.logger.info(f"✅ {workflow_name} workflow completed successfully")

            except Exception as e:
                self.logger.error(f"❌ {workflow_name} workflow failed: {e}")

            # Pause between workflows
            self.logger.info("\nPausing before next workflow...")
            await asyncio.sleep(5)

        self.logger.info("\n🎉 All workflow examples completed!")


async def main():
    """Main function to run workflow examples."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        workflow_name = sys.argv[1]
    else:
        workflow_name = "all"

    # Create and run workflow examples
    workflows = WorkflowExamples()

    try:
        await workflows.initialize()

        if workflow_name == "all":
            await workflows.run_all_workflows()
        else:
            await workflows.run_workflow(workflow_name)

    except KeyboardInterrupt:
        print("\n⏹️ Workflow examples interrupted by user")
    except Exception as e:
        print(f"❌ Workflow examples failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await workflows.cleanup()


if __name__ == "__main__":
    # Print usage information
    print("🔄 Multi-Agent Workflow Examples")
    print("=================================")
    print()

    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print(__doc__)
        sys.exit(0)

    # Available workflows summary
    print("Available Workflows:")
    print("📅 daily_briefing - Morning briefing with weather, news, schedule")
    print("📚 research_report - Research topic and create comprehensive report")
    print("💰 financial_analysis - Analyze finances and create investment plan")
    print("📋 project_planning - Plan project with timeline and task organization")
    print("📊 data_processing - Process multi-source data through analysis pipeline")
    print("✈️ travel_planning - Plan travel with research, weather, and budgeting")
    print()

    # Run workflow examples
    asyncio.run(main())
