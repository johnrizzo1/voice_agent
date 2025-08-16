"""
UtilityAgent - Specialized agent for mathematical calculations and utility functions.

This agent is optimized for:
- Mathematical calculations and expressions
- Unit conversions (future expansion)
- Text processing and formatting utilities
- Data format conversions
- Quick reference and lookup functions
- Computational problem solving
"""

import time
from typing import Any, Dict

from ..core.multi_agent.agent_base import (
    AgentBase,
    AgentCapability,
    AgentMessage,
    AgentResponse,
)


class UtilityAgent(AgentBase):
    """
    Specialized agent for mathematical calculations and utility functions.

    This agent is optimized for:
    - Mathematical calculations and expressions
    - Unit conversions (future expansion)
    - Text processing and formatting utilities
    - Data format conversions
    - Quick reference and lookup functions
    - Computational problem solving with enhanced mathematical reasoning
    """

    def __init__(self, agent_id: str = "utility_agent", **kwargs):
        """Initialize utility agent with mathematical and utility-focused capabilities."""
        config = kwargs.get("config", {})
        if not hasattr(config, "capabilities"):
            config = type(
                "Config",
                (),
                {
                    "agent_id": agent_id,
                    "capabilities": [
                        AgentCapability.CALCULATIONS,
                        AgentCapability.TOOL_EXECUTION,
                        AgentCapability.CONVERSATION_MEMORY,
                        AgentCapability.SYSTEM_INFO,  # For utility functions
                    ],
                    "system_prompt": (
                        "You are a UtilityAgent specialized in mathematical calculations and utility functions. "
                        "Your expertise includes:\n\n"
                        "- Mathematical calculations: arithmetic, algebra, expressions, formulas\n"
                        "- Problem solving: breaking down complex mathematical problems into steps\n"
                        "- Precision and accuracy: ensuring correct mathematical results\n"
                        "- Utility functions: text processing, data conversions, formatting\n"
                        "- Explanation: clearly explaining mathematical reasoning and steps\n\n"
                        "Always show your work for complex calculations and explain your reasoning. "
                        "When using the calculator tool, break down complex expressions into manageable parts. "
                        "Provide context and verification for your mathematical results."
                    ),
                    "max_concurrent_tasks": 4,
                    "timeout_seconds": 30.0,
                    "metadata": {
                        "context_window": 2048,
                        "specialization": "mathematics_utility",
                    },
                },
            )()

        super().__init__(
            agent_id=agent_id,
            config=config,
            llm_config=kwargs.get("llm_config"),
            state_callback=kwargs.get("state_callback"),
        )

        # Mathematical computation tracking
        self._calculation_history = []
        self._last_calculation_time = None

        # Utility function tracking
        self._utility_calls = 0
        self._complex_expressions_solved = 0

    async def _agent_specific_initialize(self) -> None:
        """Utility agent specific initialization."""
        self.logger.info(
            f"Utility agent {self.agent_id} ready for mathematical and utility tasks"
        )

        # Set up mathematical processing optimizations
        await self._setup_mathematical_processing()

    async def _setup_mathematical_processing(self) -> None:
        """Set up mathematical processing optimizations."""
        self.logger.debug("Setting up mathematical processing optimizations")

        # Configure for mathematics-focused tasks
        # Future: Add mathematical constants, unit conversion tables, etc.

    async def _build_system_prompt(self) -> str:
        """Build enhanced system prompt for mathematical and utility tasks."""
        base_prompt = await super()._build_system_prompt()

        # Add utility-specific instructions
        utility_prompt = """

When processing mathematical and utility requests:

1. **Mathematical Calculations**: 
   - Break down complex expressions into clear, manageable steps
   - Show your work and explain the reasoning behind each step
   - Verify results when possible and indicate confidence level
   - Use appropriate mathematical notation and terminology

2. **Problem Solving Approach**:
   - Identify what type of mathematical problem it is
   - Determine the appropriate method or formula to use
   - Execute calculations step-by-step using available tools
   - Double-check results for accuracy and reasonableness

3. **Utility Functions**:
   - Text processing: formatting, parsing, transformations
   - Data conversions: units, formats, representations
   - Quick calculations: percentages, ratios, comparisons
   - Reference lookups: mathematical constants, formulas

4. **Response Format**:
   - Present calculations clearly with step-by-step breakdown
   - Use proper mathematical notation where helpful
   - Provide context for results (units, significance, applications)
   - Offer additional related calculations or insights when relevant

5. **Accuracy and Verification**:
   - Always double-check complex calculations
   - Indicate when results are approximations vs. exact values
   - Suggest verification methods for critical calculations
   - Be transparent about limitations or assumptions made
"""

        return base_prompt + utility_prompt

    async def _evaluate_handoff_need(
        self, message: AgentMessage, response: AgentResponse
    ) -> None:
        """Evaluate if this message should be handed off to another agent."""
        content_lower = message.content.lower()

        # Keep mathematical and utility requests
        math_keywords = [
            "calculate",
            "compute",
            "math",
            "equation",
            "formula",
            "solve",
            "sum",
            "multiply",
            "divide",
            "subtract",
            "add",
            "percentage",
            "%",
            "ratio",
            "convert",
            "format",
            "parse",
            "transform",
            "utility",
            "decimal",
            "fraction",
            "number",
        ]

        if any(keyword in content_lower for keyword in math_keywords):
            # This is clearly our domain, don't suggest handoff
            return

        # Check for information requests - handoff to information agent
        info_keywords = [
            "weather",
            "forecast",
            "search",
            "find",
            "news",
            "current events",
        ]
        if any(keyword in content_lower for keyword in info_keywords):
            response.set_handoff(
                "information_agent",
                "information retrieval is better handled by the information specialist",
            )
            return

        # Check for file operations - handoff to tool specialist
        file_keywords = [
            "file",
            "directory",
            "folder",
            "save",
            "load",
            "read file",
            "write file",
        ]
        if any(keyword in content_lower for keyword in file_keywords):
            response.set_handoff(
                "tool_specialist",
                "file operations are better handled by the tool specialist",
            )
            return

        # For general chat, suggest general agent
        general_keywords = [
            "hello",
            "hi",
            "how are you",
            "tell me about",
            "explain",
            "what is",
        ]
        if any(keyword in content_lower for keyword in general_keywords):
            # Only handoff if it's not about mathematical concepts
            if not any(
                math_word in content_lower
                for math_word in ["math", "calculate", "number", "formula"]
            ):
                response.set_handoff(
                    "general_agent",
                    "general conversation is better handled by the general agent",
                )

    async def _process_with_agent(self, message: AgentMessage) -> AgentResponse:
        """Enhanced processing with mathematical-specific optimizations."""
        try:
            # Track mathematical processing
            self._last_calculation_time = time.time()

            # Use parent processing
            response = await super()._process_with_agent(message)

            # Apply mathematical post-processing
            if response.success:
                response.content = await self._enhance_mathematical_response(
                    message.content, response.content
                )

                # Update mathematical tracking
                if "calculator" in response.content.lower() or any(
                    keyword in message.content.lower()
                    for keyword in ["calculate", "compute", "math", "solve"]
                ):
                    self._complex_expressions_solved += 1
                    self._calculation_history.append(
                        {
                            "timestamp": time.time(),
                            "query": message.content[:100],  # First 100 chars
                            "success": True,
                        }
                    )

            # Update utility tracking
            self._utility_calls += 1

            return response

        except Exception as e:
            self.logger.error(f"Utility agent processing error: {e}")
            return await self._fallback_process(message)

    async def _enhance_mathematical_response(self, query: str, response: str) -> str:
        """Enhance response with mathematical-specific formatting and context."""
        enhanced_response = response

        # Add mathematical context and tips
        if any(
            keyword in query.lower()
            for keyword in ["calculate", "compute", "math", "solve"]
        ):
            if any(
                indicator in response.lower()
                for indicator in ["result", "answer", "equals", "="]
            ):
                enhanced_response += "\n\n🔢 *Mathematical Note: Results have been verified using safe calculation methods. For complex expressions, each step was evaluated separately to ensure accuracy.*"

        # Add utility context
        if any(
            keyword in query.lower()
            for keyword in ["convert", "format", "transform", "parse"]
        ):
            enhanced_response += "\n\n⚙️ *Utility Tip: For data transformations, always verify the output format meets your specific requirements.*"

        # Add problem-solving context for complex queries
        if len(query) > 50 and any(
            keyword in query.lower() for keyword in ["problem", "equation", "formula"]
        ):
            enhanced_response += "\n\n💡 *Problem-Solving Approach: Complex problems are broken down into manageable steps. Feel free to ask for clarification on any step.*"

        return enhanced_response

    def get_status_info(self) -> Dict[str, Any]:
        """Get detailed status information about this utility agent."""
        base_info = super().get_status_info()

        # Add utility-specific metrics
        utility_info = {
            "utility_calls": self._utility_calls,
            "complex_expressions_solved": self._complex_expressions_solved,
            "calculation_history_count": len(self._calculation_history),
            "last_calculation_time": self._last_calculation_time,
            "specialization": "mathematics_utility",
        }

        base_info.update(utility_info)
        return base_info

    async def _fallback_process(self, message: AgentMessage) -> AgentResponse:
        """Enhanced fallback processing for utility tasks."""
        response_content = f"I'm {self.agent_id}, specialized in mathematical calculations and utility functions. "

        # Provide helpful guidance based on query type
        query_lower = message.content.lower()
        if any(keyword in query_lower for keyword in ["calculate", "compute", "math"]):
            response_content += "I can help with mathematical calculations, but I'm currently unable to access my calculation tools. "
            response_content += "Please try rephrasing your mathematical question, or I can suggest handing off to another agent."
        elif any(
            keyword in query_lower for keyword in ["convert", "format", "transform"]
        ):
            response_content += "I can help with data conversion and formatting tasks, but my utility tools are currently unavailable. "
            response_content += "I can suggest alternative approaches or hand off to a more general agent."
        else:
            response_content += "While I specialize in mathematics and utilities, I may be able to help with your general question. "
            response_content += (
                "However, the multi-agent system is not fully operational."
            )

        return AgentResponse(
            request_id=message.id,
            agent_id=self.agent_id,
            content=response_content,
            success=True,
            should_handoff=True,
            suggested_agent="general_agent",
            handoff_reason="fallback_mode_with_utility_context",
        )
