"""
Custom tool example for the voice agent.

This example demonstrates how to create and register custom tools
for extending the voice agent's capabilities.
"""

import asyncio
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import psutil
from pydantic import BaseModel, Field

from voice_agent import VoiceAgent
from voice_agent.core.config import Config
from voice_agent.tools.base import Tool
from voice_agent.tools.registry import register_tool, tool


# Example 1: Custom tool using the Tool base class
class SystemInfoParameters(BaseModel):
    """Parameters for the system info tool."""

    include_processes: bool = Field(
        default=False, description="Include running processes information"
    )
    max_processes: int = Field(
        default=5, description="Maximum number of processes to show"
    )


class SystemInfoTool(Tool):
    """
    Custom tool to get system information.

    Demonstrates how to create a tool class with proper parameter validation
    and comprehensive system information gathering.
    """

    name = "system_info"
    description = (
        "Get detailed system information including CPU, memory, and disk usage"
    )
    version = "1.0.0"

    Parameters = SystemInfoParameters

    def __init__(self):
        """Initialize the system info tool."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize the tool."""
        await super().initialize()
        self.logger.info("System info tool initialized")

    def execute(
        self, include_processes: bool = False, max_processes: int = 5
    ) -> Dict[str, Any]:
        """
        Get system information.

        Args:
            include_processes: Whether to include process information
            max_processes: Maximum number of processes to show

        Returns:
            Dictionary containing system information
        """
        try:
            # Get basic system information
            system_info = {
                "timestamp": datetime.now().isoformat(),
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                },
                "cpu": {
                    "usage_percent": psutil.cpu_percent(interval=1),
                    "count_logical": psutil.cpu_count(logical=True),
                    "count_physical": psutil.cpu_count(logical=False),
                    "frequency": (
                        dict(psutil.cpu_freq()._asdict()) if psutil.cpu_freq() else None
                    ),
                },
                "memory": {
                    "virtual": dict(psutil.virtual_memory()._asdict()),
                    "swap": dict(psutil.swap_memory()._asdict()),
                },
                "disk": {
                    "usage": dict(psutil.disk_usage("/")._asdict()),
                    "io": (
                        dict(psutil.disk_io_counters()._asdict())
                        if psutil.disk_io_counters()
                        else None
                    ),
                },
                "network": {
                    "io": (
                        dict(psutil.net_io_counters()._asdict())
                        if psutil.net_io_counters()
                        else None
                    )
                },
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            }

            # Add process information if requested
            if include_processes:
                processes = []
                for proc in psutil.process_iter(
                    ["pid", "name", "cpu_percent", "memory_percent"]
                ):
                    try:
                        proc_info = proc.info
                        proc_info["cpu_percent"] = proc.cpu_percent()
                        processes.append(proc_info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                # Sort by CPU usage and limit results
                processes.sort(key=lambda x: x.get("cpu_percent", 0), reverse=True)
                system_info["top_processes"] = processes[:max_processes]

            return {"success": True, "result": system_info, "error": None}

        except Exception as e:
            self.logger.error(f"System info error: {e}")
            return {"success": False, "result": None, "error": str(e)}

    async def cleanup(self) -> None:
        """Cleanup the tool."""
        self.logger.info("System info tool cleaned up")
        await super().cleanup()


# Example 2: Simple tool using the @tool decorator
@tool(name="timestamp", description="Get current timestamp in various formats")
def get_timestamp(format_type: str = "iso") -> Dict[str, Any]:
    """
    Get current timestamp.

    Args:
        format_type: Format type (iso, unix, readable)

    Returns:
        Timestamp information
    """
    now = datetime.now()

    formats = {
        "iso": now.isoformat(),
        "unix": now.timestamp(),
        "readable": now.strftime("%Y-%m-%d %H:%M:%S"),
        "utc": datetime.utcnow().isoformat() + "Z",
    }

    if format_type in formats:
        return {
            "format": format_type,
            "timestamp": formats[format_type],
            "all_formats": formats,
        }
    else:
        raise ValueError(
            f"Unknown format type: {format_type}. Available: {list(formats.keys())}"
        )


# Example 3: Tool with more complex parameters
class NetworkToolParameters(BaseModel):
    """Parameters for the network tool."""

    operation: str = Field(description="Operation: 'ping', 'resolve', 'ports'")
    target: str = Field(description="Target hostname or IP address")
    count: int = Field(default=4, description="Number of ping attempts")


@tool(
    name="network_check",
    description="Perform basic network operations",
    parameters=NetworkToolParameters,
)
def network_check(operation: str, target: str, count: int = 4) -> Dict[str, Any]:
    """
    Perform network operations (placeholder implementation).

    Args:
        operation: Operation to perform
        target: Target host
        count: Number of attempts

    Returns:
        Network operation results
    """
    # This is a placeholder implementation
    # In a real implementation, you would use subprocess or libraries like ping3

    results = {
        "operation": operation,
        "target": target,
        "timestamp": datetime.now().isoformat(),
    }

    if operation == "ping":
        # Mock ping results
        results.update(
            {
                "success": True,
                "count": count,
                "avg_response_time": "10.5ms",
                "packet_loss": "0%",
                "note": "This is a mock implementation",
            }
        )
    elif operation == "resolve":
        # Mock DNS resolution
        results.update(
            {"resolved_ip": "192.168.1.100", "note": "This is a mock implementation"}
        )
    elif operation == "ports":
        # Mock port scan
        results.update(
            {"open_ports": [80, 443, 22], "note": "This is a mock implementation"}
        )
    else:
        raise ValueError(f"Unknown operation: {operation}")

    return results


async def demo_custom_tools():
    """Demonstrate custom tools with the voice agent."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting custom tool demonstration...")

    try:
        # Register the custom tool class
        system_tool = SystemInfoTool()
        register_tool(system_tool)

        logger.info("Custom tools registered")

        # Load configuration and create voice agent
        config_path = Path(__file__).parent.parent / "config" / "default.yaml"
        config = Config.load(config_path)

        # Enable our custom tools in config
        config.tools.enabled.extend(["system_info", "timestamp", "network_check"])

        agent = VoiceAgent(config=config)
        await agent.initialize()

        logger.info("Voice agent initialized with custom tools")

        # Test custom tools with text queries
        test_queries = [
            "Get system information",
            "What's the current timestamp?",
            "Get timestamp in readable format",
            "Check network ping to google.com",
            "Get system info with process details",
        ]

        logger.info("Testing custom tools...")

        for query in test_queries:
            logger.info(f"\n--- Testing: '{query}' ---")

            try:
                response = await agent.process_text(query)
                logger.info(f"Response: {response}")
            except Exception as e:
                logger.error(f"Error processing query: {e}")

            await asyncio.sleep(1)

        # Test tools directly
        await test_tools_directly()

    except Exception as e:
        logger.error(f"Error in custom tool demo: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "agent" in locals():
            await agent.stop()
        logger.info("Custom tool demonstration completed")


async def test_tools_directly():
    """Test custom tools directly without the voice agent."""

    logger = logging.getLogger(__name__)
    logger.info("\n=== Direct Tool Testing ===")

    # Test SystemInfoTool
    logger.info("\n--- SystemInfoTool ---")
    system_tool = SystemInfoTool()
    await system_tool.initialize()

    result = system_tool.execute(include_processes=True, max_processes=3)
    if result["success"]:
        info = result["result"]
        logger.info(
            f"  Platform: {info['platform']['system']} {info['platform']['release']}"
        )
        logger.info(f"  CPU Usage: {info['cpu']['usage_percent']}%")
        logger.info(f"  Memory Usage: {info['memory']['virtual']['percent']}%")
        if "top_processes" in info:
            logger.info(f"  Top processes: {len(info['top_processes'])} shown")
    else:
        logger.error(f"  Error: {result['error']}")

    await system_tool.cleanup()

    # Test timestamp tool
    logger.info("\n--- Timestamp Tool ---")
    try:
        ts_result = get_timestamp("readable")
        logger.info(f"  Readable timestamp: {ts_result['timestamp']}")
    except Exception as e:
        logger.error(f"  Error: {e}")

    # Test network tool
    logger.info("\n--- Network Tool ---")
    try:
        net_result = network_check("ping", "example.com", 3)
        logger.info(f"  Ping result: {net_result['avg_response_time']}")
    except Exception as e:
        logger.error(f"  Error: {e}")


if __name__ == "__main__":
    # Run the custom tool demonstration
    asyncio.run(demo_custom_tools())
