"""
Basic chat example for the voice agent.

This example demonstrates how to set up and run a simple voice chat session
with the voice agent using default configuration.
"""

import asyncio
import logging
from pathlib import Path

from voice_agent import VoiceAgent
from voice_agent.core.config import Config


async def main():
    """Run a basic chat session with the voice agent."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting basic chat example...")

    try:
        # Load default configuration
        config_path = Path(__file__).parent.parent / "config" / "default.yaml"

        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return

        # Create voice agent
        config = Config.load(config_path)
        agent = VoiceAgent(config=config)

        logger.info("Initializing voice agent...")
        await agent.initialize()

        logger.info("Voice agent initialized successfully!")
        logger.info("Starting chat session...")
        logger.info("Speak to interact with the agent, or press Ctrl+C to exit")

        # Start the voice agent
        await agent.start()

    except KeyboardInterrupt:
        logger.info("Chat session ended by user")
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
    except Exception as e:
        logger.error(f"Error during chat session: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup
        if "agent" in locals():
            await agent.stop()
        logger.info("Basic chat example finished")


if __name__ == "__main__":
    # Run the basic chat example
    asyncio.run(main())
