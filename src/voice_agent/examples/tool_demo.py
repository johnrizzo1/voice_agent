"""
Tool demonstration example for the voice agent.

This example shows how to interact with the voice agent's built-in tools
and demonstrates various tool capabilities.
"""

import asyncio
import logging
from pathlib import Path

from voice_agent import VoiceAgent
from voice_agent.core.config import Config
from voice_agent.tools.builtin.calculator import CalculatorTool
from voice_agent.tools.builtin.file_ops import FileOpsTool
from voice_agent.tools.builtin.weather import WeatherTool


async def demo_tools():
    """Demonstrate voice agent tools without audio."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting tool demonstration...")

    try:
        # Load configuration
        config_path = Path(__file__).parent.parent / "config" / "default.yaml"
        config = Config.load(config_path)

        # Create voice agent
        agent = VoiceAgent(config=config)
        await agent.initialize()

        logger.info("Voice agent initialized for tool demonstration")

        # Test queries that will use tools
        test_queries = [
            "What is 25 * 7 + 12?",
            "Calculate the square root of 144",
            "What's the weather like in London?",
            "Get weather for New York in fahrenheit",
            "List files in the current directory",
            "Check if setup.py exists",
        ]

        logger.info("Testing tool functionality with text queries...")

        for query in test_queries:
            logger.info(f"\n--- Testing query: '{query}' ---")

            try:
                # Process text query (bypassing audio)
                response = await agent.process_text(query)
                logger.info(f"Response: {response}")

            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")

            # Small delay between queries
            await asyncio.sleep(1)

        # Demonstrate individual tools
        await demo_individual_tools()

    except Exception as e:
        logger.error(f"Error in tool demonstration: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "agent" in locals():
            await agent.stop()
        logger.info("Tool demonstration completed")


async def demo_individual_tools():
    """Demonstrate individual tools directly."""

    logger = logging.getLogger(__name__)
    logger.info("\n=== Individual Tool Demonstrations ===")

    # Calculator tool demo
    logger.info("\n--- Calculator Tool Demo ---")
    calc_tool = CalculatorTool()
    await calc_tool.initialize()

    calc_tests = [
        "2 + 3 * 4",
        "(10 + 5) / 3",
        "2 ** 8",
        "abs(-42)",
        "max(1, 2, 3, 4, 5)",
    ]

    for expression in calc_tests:
        result = calc_tool.execute(expression=expression)
        if result["success"]:
            logger.info(f"  {expression} = {result['result']}")
        else:
            logger.error(f"  {expression} -> Error: {result['error']}")

    # Weather tool demo
    logger.info("\n--- Weather Tool Demo ---")
    weather_tool = WeatherTool()
    await weather_tool.initialize()

    weather_tests = [
        ("London", "celsius"),
        ("New York", "fahrenheit"),
        ("Tokyo", "celsius"),
    ]

    for location, units in weather_tests:
        result = weather_tool.execute(location=location, units=units)
        if result["success"]:
            weather_data = result["result"]
            logger.info(f"  {location}: {weather_data['description']}")
        else:
            logger.error(f"  {location} -> Error: {result['error']}")

    # File operations tool demo
    logger.info("\n--- File Operations Tool Demo ---")
    file_tool = FileOpsTool()
    await file_tool.initialize()

    # Test file operations
    file_tests = [
        ("exists", "setup.py"),
        ("exists", "requirements.txt"),
        ("list", ".", {"recursive": False}),
        ("read", "requirements.txt"),  # Only if exists
    ]

    for operation, path, *args in file_tests:
        extra_args = args[0] if args else {}
        result = file_tool.execute(operation=operation, path=path, **extra_args)

        if result["success"]:
            if operation == "exists":
                exists = result["result"]["exists"]
                logger.info(f"  {path} exists: {exists}")
            elif operation == "list":
                items = result["result"]["items"]
                logger.info(f"  Directory listing ({len(items)} items):")
                for item in items[:5]:  # Show first 5 items
                    logger.info(f"    {item['type']}: {item['name']}")
            elif operation == "read":
                content_length = len(result["result"]["content"])
                logger.info(f"  Read {path}: {content_length} characters")
        else:
            logger.error(f"  {operation} {path} -> Error: {result['error']}")


async def interactive_tool_demo():
    """Interactive tool demonstration where user can type queries."""

    logger = logging.getLogger(__name__)
    logger.info("\n=== Interactive Tool Demo ===")
    logger.info("Type queries to test tools. Type 'quit' to exit.")

    try:
        # Load configuration
        config_path = Path(__file__).parent.parent / "config" / "default.yaml"
        config = Config.load(config_path)

        # Create voice agent
        agent = VoiceAgent(config=config)
        await agent.initialize()

        while True:
            try:
                # Get user input
                query = input("\nEnter query (or 'quit'): ").strip()

                if query.lower() in ["quit", "exit", "q"]:
                    break

                if not query:
                    continue

                # Process the query
                logger.info(f"Processing: {query}")
                response = await agent.process_text(query)
                logger.info(f"Response: {response}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")

        await agent.stop()

    except Exception as e:
        logger.error(f"Error in interactive demo: {e}")


if __name__ == "__main__":
    # Choose which demo to run
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        # Run interactive demo
        asyncio.run(interactive_tool_demo())
    else:
        # Run automated tool demo
        asyncio.run(demo_tools())
