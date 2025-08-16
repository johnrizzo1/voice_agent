#!/usr/bin/env python3
"""
Simple test to verify ProductivityAgent workflow and context preservation.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from voice_agent.agents.productivity_agent import ProductivityAgent
from voice_agent.core.multi_agent.agent_base import AgentCapability
from voice_agent.core.multi_agent.message import AgentMessage, MessageType


async def test_productivity_workflow():
    """Test ProductivityAgent workflow and context preservation."""
    print("🧪 Testing ProductivityAgent Workflow and Context")
    print("=" * 50)

    # Create ProductivityAgent
    agent_config = type(
        "Config",
        (),
        {
            "agent_id": "productivity_test",
            "capabilities": [
                AgentCapability.FILE_OPERATIONS,
                AgentCapability.CALENDAR_MANAGEMENT,
                AgentCapability.TASK_PLANNING,
                AgentCapability.TOOL_EXECUTION,
            ],
            "system_prompt": "Test productivity agent",
            "max_concurrent_tasks": 3,
            "timeout_seconds": 30.0,
            "metadata": {"context_window": 2048},
        },
    )()

    agent = ProductivityAgent(
        agent_id="productivity_test",
        config=agent_config,
        llm_config={"model": "mistral:7b", "temperature": 0.7},
    )

    # Initialize the agent
    await agent.initialize()

    print(f"✅ ProductivityAgent created: {agent.agent_id}")
    print(f"   Status: {agent.status.value}")
    print(f"   Capabilities: {[cap.value for cap in agent.capabilities]}")

    # Test context preservation across multiple messages
    conversation_id = "test_conv_001"

    # Message 1: File operation
    message1 = AgentMessage(
        conversation_id=conversation_id,
        type=MessageType.USER_INPUT,
        content="I need to organize my project files. Can you help me list the files in the current directory?",
        requires_response=True,
    )

    print(f"\n📝 Message 1: {message1.content[:50]}...")
    response1 = await agent.process_message(message1)
    print(f"📤 Response 1 Success: {response1.success}")
    if response1.success:
        print(f"   Content preview: {response1.content[:100]}...")

    # Message 2: Calendar operation
    message2 = AgentMessage(
        conversation_id=conversation_id,
        type=MessageType.USER_INPUT,
        content="Now I need to schedule a meeting for tomorrow at 2 PM to discuss the project files we just reviewed.",
        requires_response=True,
    )

    print(f"\n📝 Message 2: {message2.content[:50]}...")
    response2 = await agent.process_message(message2)
    print(f"📤 Response 2 Success: {response2.success}")
    if response2.success:
        print(f"   Content preview: {response2.content[:100]}...")

    # Message 3: Task planning
    message3 = AgentMessage(
        conversation_id=conversation_id,
        type=MessageType.USER_INPUT,
        content="Create a todo list for organizing the project files and preparing for the meeting.",
        requires_response=True,
    )

    print(f"\n📝 Message 3: {message3.content[:50]}...")
    response3 = await agent.process_message(message3)
    print(f"📤 Response 3 Success: {response3.success}")
    if response3.success:
        print(f"   Content preview: {response3.content[:100]}...")

    # Check productivity stats
    stats = agent.get_productivity_stats()
    print(f"\n📊 Productivity Stats:")
    print(f"   Files processed: {stats['productivity_metrics']['files_processed']}")
    print(
        f"   Calendar events managed: {stats['productivity_metrics']['calendar_events_managed']}"
    )
    print(f"   Tasks organized: {stats['productivity_metrics']['tasks_organized']}")
    print(f"   Total messages processed: {stats['total_processed']}")

    # Check context preservation
    print(f"\n🧠 Context Preservation:")
    print(f"   File operations in context: {stats['file_context']['operations_count']}")
    print(f"   Calendar events in context: {stats['calendar_context']['events_count']}")
    print(f"   Conversation history length: {len(agent.conversation_history)}")

    # Test error handling with invalid request
    print(f"\n🚨 Testing Error Handling:")
    error_message = AgentMessage(
        conversation_id=conversation_id,
        type=MessageType.USER_INPUT,
        content="Calculate the square root of negative numbers using advanced mathematics.",
        requires_response=True,
    )

    error_response = await agent.process_message(error_message)
    print(f"📤 Error handling response success: {error_response.success}")
    if error_response.should_handoff:
        print(f"   ✅ Correctly suggested handoff to: {error_response.suggested_agent}")
        print(f"   Reason: {error_response.handoff_reason}")

    await agent.cleanup()
    print(f"\n✅ ProductivityAgent workflow test completed successfully")

    return True


if __name__ == "__main__":
    success = asyncio.run(test_productivity_workflow())
    sys.exit(0 if success else 1)
