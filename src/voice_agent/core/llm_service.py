"""
Language Model service for the voice agent.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from .config import LLMConfig

try:
    import ollama
except ImportError:
    ollama = None

try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    transformers = None
    AutoTokenizer = None
    AutoModelForCausalLM = None


class LLMService:
    """
    Language Model service supporting multiple backends.

    Supports:
    - Ollama (primary, local model serving)
    - Transformers (direct model loading)
    - Function calling capabilities
    """

    def __init__(
        self,
        config: LLMConfig,
        state_callback: Optional[Callable[[str, str, Optional[str]], None]] = None,
    ):
        """
        Initialize the LLM service.

        Args:
            config: LLM configuration settings
            state_callback: Optional callback(component, state, message)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._state_callback = state_callback

        # Model instances
        self.ollama_client: Optional[ollama.Client] = None
        self.transformers_model: Optional[Any] = None
        self.transformers_tokenizer: Optional[Any] = None

        # Service state
        self.is_initialized = False
        self.current_backend = self._determine_backend()

        # Function calling
        self.available_functions: Dict[str, Callable] = {}

    def _determine_backend(self) -> str:
        """Determine which LLM backend to use based on availability and config."""
        if self.config.provider == "ollama" and ollama:
            return "ollama"
        elif self.config.provider == "transformers" and transformers:
            return "transformers"
        elif ollama:
            return "ollama"
        elif transformers:
            return "transformers"
        else:
            self.logger.error("No LLM backend available")
            return "none"

    def set_state_callback(
        self, cb: Optional[Callable[[str, str, Optional[str]], None]]
    ) -> None:
        self._state_callback = cb

    def _emit_state(self, state: str, message: Optional[str] = None) -> None:
        if self._state_callback:
            try:
                self._state_callback("llm", state, message)
            except Exception:
                self.logger.debug("LLM state callback error", exc_info=True)

    async def initialize(self) -> None:
        """Initialize the LLM service and load models."""
        self.logger.info(
            f"Initializing LLM service with backend: {self.current_backend}"
        )
        self._emit_state("initializing", f"backend={self.current_backend}")

        if self.current_backend == "ollama":
            await self._initialize_ollama()
        elif self.current_backend == "transformers":
            await self._initialize_transformers()
        else:
            self.logger.error("No LLM backend could be initialized")
            return

        self.is_initialized = True
        self.logger.info("LLM service initialized")
        self._emit_state("ready", None)

    async def _initialize_ollama(self) -> None:
        """Initialize Ollama client."""
        try:
            # Create Ollama client
            self.ollama_client = ollama.Client()

            # Check if model is available
            model_name = self.config.model

            # Try to pull model if not available
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.ollama_client.show(model_name)
                )
                self.logger.info(f"Ollama model '{model_name}' is available")
            except Exception:
                self.logger.info(f"Pulling Ollama model '{model_name}'...")
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.ollama_client.pull(model_name)
                )
                self.logger.info(f"Ollama model '{model_name}' pulled successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama: {e}")
            self.ollama_client = None

    async def _initialize_transformers(self) -> None:
        """Initialize Transformers model."""
        try:
            model_name = self.config.model

            # Load model and tokenizer in separate thread
            loop = asyncio.get_event_loop()

            self.transformers_tokenizer = await loop.run_in_executor(
                None, lambda: AutoTokenizer.from_pretrained(model_name)
            )

            self.transformers_model = await loop.run_in_executor(
                None,
                lambda: AutoModelForCausalLM.from_pretrained(
                    model_name, device_map="auto", torch_dtype="auto"
                ),
            )

            self.logger.info(f"Transformers model '{model_name}' loaded")

        except Exception as e:
            self.logger.error(f"Failed to initialize Transformers: {e}")
            self.transformers_model = None
            self.transformers_tokenizer = None

    async def generate_response(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]],
        tool_executor: Optional[Any] = None,
    ) -> str:
        """
        Generate a response to user input.

        Args:
            user_input: User's input text
            conversation_history: Previous conversation messages
            tool_executor: Tool executor for function calling

        Returns:
            Generated response text
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            # Prepare messages
            messages = self._prepare_messages(user_input, conversation_history)
            self._emit_state("active", "generating")

            # Check if we need to use tools
            if tool_executor and self._should_use_tools(user_input):
                result = await self._generate_with_tools(messages, tool_executor)
            else:
                result = await self._generate_simple(messages)
            self._emit_state("ready", None)
            return result

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            self._emit_state("error", "generation error")
            return (
                "I apologize, but I encountered an error while processing your request."
            )

    def _prepare_messages(
        self, user_input: str, conversation_history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Prepare messages for the model."""
        messages = []

        # Add system message
        messages.append(
            {
                "role": "system",
                "content": "You are a helpful voice assistant. Respond concisely and naturally.",
            }
        )

        # Add conversation history (keep within context window)
        max_history = min(len(conversation_history), 10)  # Limit history
        if max_history > 0:
            messages.extend(conversation_history[-max_history:])

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        return messages

    def _should_use_tools(self, user_input: str) -> bool:
        """Determine if the user input requires tool usage."""
        # Simple heuristics - could be improved with better classification
        tool_keywords = [
            "calculate",
            "compute",
            "math",
            "weather",
            "temperature",
            "file",
            "read",
            "write",
            "search",
            "lookup",
            "find",
        ]

        return any(keyword in user_input.lower() for keyword in tool_keywords)

    async def _generate_simple(self, messages: List[Dict[str, str]]) -> str:
        """Generate a simple response without tools."""
        if self.current_backend == "ollama":
            return await self._generate_ollama(messages)
        elif self.current_backend == "transformers":
            return await self._generate_transformers(messages)
        else:
            return "I'm sorry, I'm not properly configured to respond right now."

    async def _generate_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using Ollama."""
        if not self.ollama_client:
            return "Ollama service is not available."

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.ollama_client.chat(
                    model=self.config.model,
                    messages=messages,
                    options={
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    },
                ),
            )

            return response["message"]["content"].strip()

        except Exception as e:
            self.logger.error(f"Ollama generation error: {e}")
            return "I encountered an error while processing your request."

    async def _generate_transformers(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using Transformers."""
        if not self.transformers_model or not self.transformers_tokenizer:
            return "Transformers service is not available."

        try:
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)

            loop = asyncio.get_event_loop()

            # Tokenize
            inputs = await loop.run_in_executor(
                None,
                lambda: self.transformers_tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=2048
                ),
            )

            # Generate
            outputs = await loop.run_in_executor(
                None,
                lambda: self.transformers_model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.transformers_tokenizer.eos_token_id,
                ),
            )

            # Decode response
            response = await loop.run_in_executor(
                None,
                lambda: self.transformers_tokenizer.decode(
                    outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
                ),
            )

            return response.strip()

        except Exception as e:
            self.logger.error(f"Transformers generation error: {e}")
            return "I encountered an error while processing your request."

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a single prompt string."""
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"

        prompt += "Assistant: "
        return prompt

    async def _generate_with_tools(
        self, messages: List[Dict[str, str]], tool_executor: Any
    ) -> str:
        """Generate response with tool calling capabilities."""
        try:
            # Get available tools
            available_tools = await tool_executor.get_available_tools()

            # Add tool information to system message
            if available_tools:
                tool_info = self._format_tool_info(available_tools)
                messages[0]["content"] += f"\n\nAvailable tools:\n{tool_info}"
                messages[0][
                    "content"
                ] += "\n\nIf you need to use a tool, format your response as: TOOL_CALL: tool_name(arg1='value1', arg2='value2')"

            # Generate initial response
            response = await self._generate_simple(messages)

            # Check (case-insensitive) if response contains ANY *_CALL: pattern (e.g. TOOL_CALL:, WEATHER_CALL:)
            import re as _re_check

            if _re_check.search(r"[A-Z_]+_CALL:", response.upper()):
                return await self._handle_tool_call(response, tool_executor)

            # Fallback: attempt to interpret a bare function-style invocation like: weather(location="Paris") or file_ops(operation="exists", path="README.md")
            direct = await self._try_direct_tool_invocation(response, tool_executor)
            if direct is not None:
                return direct

            return response

        except Exception as e:
            self.logger.error(f"Error in tool-enabled generation: {e}")
            return await self._generate_simple(messages)

    def _format_tool_info(self, tools: List[Dict[str, Any]]) -> str:
        """Format tool information for the model."""
        tool_descriptions = []
        for tool in tools:
            name = tool.get("name", "unknown")
            description = tool.get("description", "No description")
            parameters = tool.get("parameters", {})

            param_str = ", ".join(
                [f"{k}: {v.get('type', 'any')}" for k, v in parameters.items()]
            )

            tool_descriptions.append(f"- {name}({param_str}): {description}")

        return "\n".join(tool_descriptions)

    async def _try_direct_tool_invocation(
        self, text: str, tool_executor: Any
    ) -> Optional[str]:
        """
        Attempt to parse and execute a direct function style tool invocation without *_CALL: tag.

        Examples accepted:
          weather(location="London", units="F")
          file_ops(operation="exists", path="README.md")
        """
        try:
            import re as _re

            stripped = text.strip()
            # Heuristic: single line, looks like func(...)
            if "\n" in stripped:
                return None
            m = _re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$", stripped)
            if not m:
                return None
            tool_name = m.group(1).lower()
            params_inside = m.group(2).strip()
            parameters = self._parse_tool_parameters(params_inside)

            exec_result = await tool_executor.execute_tool(tool_name, parameters)
            if not exec_result.get("success"):
                return f"The '{tool_name}' tool failed: {exec_result.get('error', 'unknown error')}"
            result = exec_result.get("result")

            # Reuse existing formatting logic by temporarily faking the structures
            if tool_name == "weather":
                return self._format_weather_response(
                    result if isinstance(result, dict) else {"description": str(result)}
                )
            if isinstance(result, dict):
                desc = result.get("description")
                filtered = [(k, v) for k, v in result.items() if k != "description"][
                    :12
                ]
                parts: List[str] = []
                if desc:
                    parts.append(desc)
                if filtered:
                    parts.append("; ".join(f"{k}={v}" for k, v in filtered))
                return " ".join(parts) if parts else "Tool executed."
            if isinstance(result, (list, tuple)):
                if not result:
                    return "No results."
                preview = "; ".join(map(str, list(result)[:10]))
                if len(result) > 10:
                    preview += f"; ... (+{len(result)-10} more)"
                return preview
            return str(result)
        except Exception:
            return None

    async def _handle_tool_call(self, response: str, tool_executor: Any) -> str:
        """Handle tool calling (robust + case-insensitive) with enhanced visualization."""
        try:
            upper_resp = response.upper()

            # Accept generic pattern e.g. TOOL_CALL:, WEATHER_CALL:, SEARCH_CALL:, etc.
            import re as _re

            m_tag = _re.search(r"([A-Z_]+)_CALL:", upper_resp)
            if not m_tag:
                return response
            idx = m_tag.start()
            tag_len = len(m_tag.group(0))
            tool_call_part = response[idx + tag_len :].strip()
            tool_call_part = " ".join(tool_call_part.split())

            import re

            m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*(\((.*)\))?", tool_call_part)
            if not m:
                return "I attempted to call a tool but the request format was unclear."

            tool_name = m.group(1).lower()
            params_inside = m.group(3) or ""
            parameters = self._parse_tool_parameters(params_inside)

            if (
                tool_name == "weather"
                and "location" not in parameters
                and params_inside
            ):
                stripped = params_inside.strip()
                if (stripped.startswith("'") and stripped.endswith("'")) or (
                    stripped.startswith('"') and stripped.endswith('"')
                ):
                    parameters["location"] = stripped[1:-1]

            try:
                exec_result = await tool_executor.execute_tool(tool_name, parameters)
            except Exception as exec_err:
                self.logger.error(
                    f"Tool execution exception for {tool_name}: {exec_err}"
                )
                return f"I tried to use the '{tool_name}' tool but it crashed unexpectedly."

            if not exec_result.get("success"):
                err = exec_result.get("error", "unknown error")
                return f"The '{tool_name}' tool failed: {err}"

            tool_result = exec_result.get("result")

            # Specialized formatting (weather may wrap actual payload under 'result')
            if tool_name == "weather":
                weather_payload = tool_result
                if (
                    isinstance(weather_payload, dict)
                    and "result" in weather_payload
                    and isinstance(weather_payload["result"], dict)
                ):
                    inner = weather_payload["result"]
                    # Promote outer location / units if missing
                    if "location" in weather_payload and "location" not in inner:
                        inner["location"] = weather_payload["location"]
                    if "units" in weather_payload and "temperature_unit" not in inner:
                        units_val = weather_payload["units"].lower()
                        if units_val.startswith("f"):
                            inner["temperature_unit"] = "°F"
                        elif units_val.startswith("c"):
                            inner["temperature_unit"] = "°C"
                    weather_payload = inner
                return self._format_weather_response(weather_payload)

            # Natural language formatting for dict results
            if isinstance(tool_result, dict):
                description = tool_result.get("description")
                # Remove very verbose or internal keys if any in future
                filtered_items = [
                    (k, v) for k, v in tool_result.items() if k not in {"description"}
                ][:12]

                if description and not filtered_items:
                    return description

                parts: List[str] = []
                if description:
                    parts.append(description)

                if filtered_items:
                    summary_pairs = "; ".join(f"{k}={v}" for k, v in filtered_items)
                    parts.append(summary_pairs)

                return (
                    " ".join(parts)
                    if parts
                    else "Tool completed with no additional details."
                )

            # Lists / tuples → concise enumeration (trim if long)
            if isinstance(tool_result, (list, tuple)):
                if not tool_result:
                    return "No results were returned."
                preview_items = list(tool_result)[:10]
                rendered = "; ".join(str(item) for item in preview_items)
                if len(tool_result) > 10:
                    rendered += f"; ... (+{len(tool_result) - 10} more)"
                return rendered

            # Fallback to string conversion
            return str(tool_result)

        except Exception as e:
            self.logger.error(f"Tool call handling error: {e}")
            return "I attempted to use a tool but encountered a parsing error."

    def _parse_tool_parameters(self, params_str: str) -> Dict[str, Any]:
        """Parse tool parameters from string format."""
        parameters = {}

        if not params_str.strip():
            return parameters

        try:
            # Simple parameter parsing for key='value' format
            import re

            # Find all key='value' or key="value" patterns
            param_pattern = r"(\w+)=(['\"])([^'\"]*)\2"
            matches = re.findall(param_pattern, params_str)

            for match in matches:
                key, quote, value = match
                parameters[key] = value

            # Also try to extract location from simple format like weather('London')
            if (
                not parameters
                and params_str.strip().startswith("'")
                and params_str.strip().endswith("'")
            ):
                # Single quoted parameter, assume it's location for weather
                location = params_str.strip()[1:-1]
                parameters["location"] = location
            elif (
                not parameters
                and params_str.strip().startswith('"')
                and params_str.strip().endswith('"')
            ):
                # Double quoted parameter
                location = params_str.strip()[1:-1]
                parameters["location"] = location

        except Exception as e:
            self.logger.debug(f"Parameter parsing error: {e}")

        return parameters

    def _format_weather_response(self, weather_data: Dict[str, Any]) -> str:
        """Format weather tool response for natural speech."""
        try:
            if isinstance(weather_data, dict) and "description" in weather_data:
                return weather_data["description"]
            elif isinstance(weather_data, dict):
                location = weather_data.get("location", "the requested location")
                temp = weather_data.get("temperature", "unknown")
                temp_unit = weather_data.get("temperature_unit", "")
                condition = weather_data.get("condition", "unknown")

                return f"The current weather in {location} is {condition} with a temperature of {temp}{temp_unit}."
            else:
                return str(weather_data)
        except Exception as e:
            self.logger.error(f"Weather response formatting error: {e}")
            return str(weather_data)

    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary containing model information
        """
        info = {
            "backend": self.current_backend,
            "provider": self.config.provider,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "initialized": self.is_initialized,
        }

        if self.current_backend == "ollama" and self.ollama_client:
            try:
                model_info = self.ollama_client.show(self.config.model)
                info["model_size"] = model_info.get("size", "unknown")
                info["model_family"] = model_info.get("family", "unknown")
            except Exception:
                pass

        return info

    def register_function(self, name: str, func: Callable) -> None:
        """
        Register a function for function calling.

        Args:
            name: Function name
            func: Function to register
        """
        self.available_functions[name] = func
        self.logger.info(f"Registered function: {name}")

    def unregister_function(self, name: str) -> None:
        """
        Unregister a function.

        Args:
            name: Function name to unregister
        """
        if name in self.available_functions:
            del self.available_functions[name]
            self.logger.info(f"Unregistered function: {name}")

    async def cleanup(self) -> None:
        """Cleanup LLM resources."""
        self.logger.info("Cleaning up LLM service...")

        self.ollama_client = None
        self.transformers_model = None
        self.transformers_tokenizer = None
        self.available_functions.clear()
        self.is_initialized = False

        self.logger.info("LLM service cleanup complete")
