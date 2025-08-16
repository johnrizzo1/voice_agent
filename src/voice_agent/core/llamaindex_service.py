"""
LlamaIndex service for the voice agent.
Provides multi-agent functionality using LlamaIndex with local Ollama models.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable

from .config import LLMConfig

try:
    from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
    from llama_index.core.agent import ReActAgent
    from llama_index.core.tools import FunctionTool
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.ollama import OllamaEmbedding

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    Settings = None
    VectorStoreIndex = None
    SimpleDirectoryReader = None
    ReActAgent = None
    FunctionTool = None
    Ollama = None
    OllamaEmbedding = None


class LlamaIndexService:
    """
    LlamaIndex service for multi-agent functionality.

    Provides:
    - Integration with existing Ollama setup
    - Agent-based conversation handling
    - Tool integration through LlamaIndex
    - Vector store capabilities for RAG
    """

    def __init__(
        self,
        config: LLMConfig,
        state_callback: Optional[Callable[[str, str, Optional[str]], None]] = None,
    ):
        """
        Initialize the LlamaIndex service.

        Args:
            config: LLM configuration settings
            state_callback: Optional callback(component, state, message)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._state_callback = state_callback

        # Service state
        self.is_initialized = False
        self.is_available = LLAMAINDEX_AVAILABLE

        # LlamaIndex components
        self.llm: Optional[Ollama] = None
        self.embedding_model: Optional[OllamaEmbedding] = None
        self.agent: Optional[ReActAgent] = None
        self.vector_index: Optional[VectorStoreIndex] = None

        # Tools registry
        self.tools: List[FunctionTool] = []

    def set_state_callback(
        self, cb: Optional[Callable[[str, str, Optional[str]], None]]
    ) -> None:
        self._state_callback = cb

    def _emit_state(self, state: str, message: Optional[str] = None) -> None:
        if self._state_callback:
            try:
                self._state_callback("llamaindex", state, message)
            except Exception:
                self.logger.debug("LlamaIndex state callback error", exc_info=True)

    async def initialize(self) -> None:
        """Initialize the LlamaIndex service."""
        if not self.is_available:
            self.logger.warning("LlamaIndex is not available - skipping initialization")
            return

        self.logger.info("Initializing LlamaIndex service with Ollama")
        self._emit_state("initializing", "setting up LlamaIndex")

        try:
            # Initialize Ollama LLM
            self.llm = Ollama(
                model=self.config.model,
                base_url="http://localhost:11434",
                temperature=self.config.temperature,
                request_timeout=60.0,
            )

            # Initialize Ollama embeddings
            self.embedding_model = OllamaEmbedding(
                model_name="nomic-embed-text",  # Good general-purpose embedding model
                base_url="http://localhost:11434",
            )

            # Configure global LlamaIndex settings
            Settings.llm = self.llm
            Settings.embed_model = self.embedding_model
            Settings.chunk_size = 512
            Settings.chunk_overlap = 50

            self.logger.info("LlamaIndex service initialized successfully")
            self.is_initialized = True
            self._emit_state("ready", None)

        except Exception as e:
            self.logger.error(f"Failed to initialize LlamaIndex service: {e}")
            self._emit_state("error", f"initialization failed: {e}")
            raise

    async def create_agent(self, tools: Optional[List[Any]] = None) -> None:
        """
        Create a ReAct agent with the provided tools.

        Args:
            tools: List of tools to provide to the agent
        """
        if not self.is_initialized:
            await self.initialize()

        if not self.is_available:
            self.logger.warning("Cannot create agent - LlamaIndex not available")
            return

        try:
            self._emit_state("active", "creating agent")

            # Convert tools if provided
            if tools:
                self.tools = self._convert_tools(tools)

            # Create ReAct agent
            self.agent = ReActAgent.from_tools(
                tools=self.tools,
                llm=self.llm,
                verbose=True,
            )

            self.logger.info(f"Created ReAct agent with {len(self.tools)} tools")
            self._emit_state("ready", f"agent created with {len(self.tools)} tools")

        except Exception as e:
            self.logger.error(f"Failed to create agent: {e}")
            self._emit_state("error", f"agent creation failed: {e}")
            raise

    def _convert_tools(self, tools: List[Any]) -> List[FunctionTool]:
        """Convert voice agent tools to LlamaIndex FunctionTool format."""
        converted_tools = []

        for tool_info in tools:
            try:
                name = tool_info.get("name", "unknown")
                description = tool_info.get("description", "No description")
                func = tool_info.get("function")

                if func:
                    function_tool = FunctionTool.from_defaults(
                        fn=func,
                        name=name,
                        description=description,
                    )
                    converted_tools.append(function_tool)
                    self.logger.debug(f"Converted tool: {name}")

            except Exception as e:
                self.logger.error(f"Failed to convert tool {tool_info}: {e}")

        return converted_tools

    async def chat(self, message: str) -> str:
        """
        Send a message to the agent and get a response.

        Args:
            message: User message

        Returns:
            Agent response
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call create_agent() first.")

        try:
            self._emit_state("active", "processing message")

            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.agent.chat(message)
            )

            self._emit_state("ready", None)
            return str(response)

        except Exception as e:
            self.logger.error(f"Chat error: {e}")
            self._emit_state("error", "chat error")
            return f"I encountered an error: {e}"

    async def create_vector_index(self, documents_path: Optional[str] = None) -> None:
        """
        Create a vector index for RAG capabilities.

        Args:
            documents_path: Path to documents directory (optional)
        """
        if not self.is_initialized:
            await self.initialize()

        if not self.is_available:
            self.logger.warning("Cannot create vector index - LlamaIndex not available")
            return

        try:
            self._emit_state("active", "creating vector index")

            if documents_path:
                # Load documents from path
                documents = SimpleDirectoryReader(documents_path).load_data()
                self.vector_index = VectorStoreIndex.from_documents(documents)
                self.logger.info(
                    f"Created vector index from {len(documents)} documents"
                )
            else:
                # Create empty index
                self.vector_index = VectorStoreIndex([])
                self.logger.info("Created empty vector index")

            self._emit_state("ready", "vector index created")

        except Exception as e:
            self.logger.error(f"Failed to create vector index: {e}")
            self._emit_state("error", f"vector index creation failed: {e}")
            raise

    async def query_vector_index(self, query: str) -> str:
        """
        Query the vector index for relevant information.

        Args:
            query: Query string

        Returns:
            Retrieved information
        """
        if not self.vector_index:
            raise RuntimeError(
                "Vector index not created. Call create_vector_index() first."
            )

        try:
            self._emit_state("active", "querying vector index")

            query_engine = self.vector_index.as_query_engine()
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: query_engine.query(query)
            )

            self._emit_state("ready", None)
            return str(response)

        except Exception as e:
            self.logger.error(f"Vector query error: {e}")
            self._emit_state("error", "vector query error")
            return f"I encountered an error querying the knowledge base: {e}"

    async def get_service_info(self) -> Dict[str, Any]:
        """Get information about the LlamaIndex service."""
        return {
            "available": self.is_available,
            "initialized": self.is_initialized,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "has_agent": self.agent is not None,
            "has_vector_index": self.vector_index is not None,
            "num_tools": len(self.tools),
        }

    async def cleanup(self) -> None:
        """Cleanup LlamaIndex resources."""
        self.logger.info("Cleaning up LlamaIndex service...")

        self.llm = None
        self.embedding_model = None
        self.agent = None
        self.vector_index = None
        self.tools.clear()
        self.is_initialized = False

        self.logger.info("LlamaIndex service cleanup complete")
