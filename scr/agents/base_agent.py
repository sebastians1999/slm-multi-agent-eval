from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Callable
from pydantic import BaseModel, Field
from openai import OpenAI
import json


class ToolCallRecord(BaseModel):
    """Record of a single tool invocation."""
    tool_name: str
    success: bool
    iteration: int
    error_message: Optional[str] = None


class agent_meta_data(BaseModel):
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    total_energy_joules: float = 0.0
    total_duration_seconds: float = 0.0
    average_watts: float = 0.0

    # Tool usage tracking
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    tool_call_count: int = 0
    unique_tools_used: List[str] = Field(default_factory=list)


class BaseAgent(ABC):

    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        tool_categories: Optional[List[str]] = ['search', 'code', 'browser'],
        tools: Optional[List[Dict]] = None,
        tool_functions: Optional[Dict[str, Callable]] = None,
        max_iterations: int = 10,
        **kwargs
    ):
        """
        Initialize the agent with dynamic tool loading.

        Args:
            model: Model name/ID to use
            temperature: Sampling temperature (0-1)
            base_url: Optional custom API base URL
            api_key: API key for authentication
            tool_categories: List of tool categories to load (e.g., ['search', 'browser'])
                           Defaults to all categories ['search', 'code', 'browser'].
                           If explicitly None, loads NO tools (empty list)
            tools: Optional custom tool schemas (bypasses category system)
            tool_functions: Optional custom tool functions (bypasses category system)
            max_iterations: Maximum tool-calling iterations
            **kwargs: Additional configuration options

        Available Categories:
            - 'search': Web search tools (Tavily)
            - 'code': Python code execution (E2B)
            - 'browser': Web browser navigation (PlayWright)

        Example:
            >>> # Load all tools (default)
            >>> agent = BaseAgent(model="gpt-4")
            >>>
            >>> # Load specific categories
            >>> agent = BaseAgent(model="gpt-4", tool_categories=['search', 'browser'])
            >>>
            >>> # Load no tools
            >>> agent = BaseAgent(model="gpt-4", tool_categories=None)
            >>>
            >>> # Use custom tools (for testing)
            >>> agent = BaseAgent(model="gpt-4", tools=[...], tool_functions={...})
        """
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.api_key = api_key
        self.config = kwargs
        self.max_iterations = max_iterations
        self.meta_data = agent_meta_data()
        self._current_iteration = 0  #  for tool usage

        # Tool loading: custom tools take precedence over categories
        if tools is not None and tool_functions is not None:
            # Custom tools provided (e.g., for testing)
            self.tools = tools
            self.tool_functions = tool_functions
            self.tool_categories = None
            self.browser_manager = None
        else:
            # Dynamic tool loading from registry
            from .tool_loader import get_tools_for_agent, get_browser_manager

            # If explicitly None, load no tools (use empty list)
            if tool_categories is None:
                tool_categories = []

            self.tool_categories = tool_categories
            self.tools, self.tool_functions = get_tools_for_agent(tool_categories)

            # Track browser manager for cleanup if browser tools loaded
            self.browser_manager = None
            if isinstance(tool_categories, list) and 'browser' in tool_categories:
                self.browser_manager = get_browser_manager()

        self.openai_client = self._create_openai_client()

    
    
    

    def _create_openai_client(self) -> OpenAI:
        """Create and return the raw OpenAI client."""
        return OpenAI(api_key=self.api_key, base_url=self.base_url)


    def _update_metadata(self, response_dict: Dict) -> None:
        """Update metadata based on API response."""
        if response_dict is None:
            return

        usage = response_dict.get("usage", {})
        if usage:
            self.meta_data.completion_tokens += usage.get("completion_tokens", 0)
            self.meta_data.prompt_tokens += usage.get("prompt_tokens", 0)
            self.meta_data.total_tokens += usage.get("total_tokens", 0)

        energy_consumption = response_dict.get("energy_consumption", {})
        if energy_consumption:
            self.meta_data.total_energy_joules += energy_consumption.get("joules", 0.0)
            self.meta_data.total_duration_seconds += energy_consumption.get("duration_seconds", 0.0)

            if self.meta_data.total_duration_seconds > 0:
                self.meta_data.average_watts = (
                    self.meta_data.total_energy_joules / self.meta_data.total_duration_seconds
                )

    def _track_tool_usage(
        self,
        tool_name: str,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """
        Track tool usage in metadata.

        Args:
            tool_name: Name of the tool that was called
            success: Whether the tool execution was successful
            error: Error message if execution failed
        """
        record = ToolCallRecord(
            tool_name=tool_name,
            success=success,
            iteration=self._current_iteration,
            error_message=error
        )
        self.meta_data.tool_calls.append(record)
        self.meta_data.tool_call_count += 1

        # Add to unique tools list if not already present
        if tool_name not in self.meta_data.unique_tools_used:
            self.meta_data.unique_tools_used.append(tool_name)

    def _execute_tool_call(self, tool_name: str, tool_args: Dict) -> str:
        """Execute a tool function and return the result."""
        if tool_name not in self.tool_functions:
            # Track failed call - tool not found
            self._track_tool_usage(tool_name, success=False, error="Tool not found")
            return f"Error: Tool '{tool_name}' not found in tool_functions"

        try:
            tool_func = self.tool_functions[tool_name]
            result = tool_func(**tool_args)
            #print(result)

            # Track successful call
            self._track_tool_usage(tool_name, success=True)

            return str(result)
        except Exception as e:
            # Track failed call - execution error
            self._track_tool_usage(tool_name, success=False, error=str(e))
            return f"Error executing tool '{tool_name}': {str(e)}"



    def invoke(self, messages: List[Dict], tools: bool = True) -> Dict:
        """
        Invoke the agent with messages. The agent will reactively decide whether to use tools.

        Args:
            messages: List of message dicts, e.g., [{"role": "user", "content": "..."}]
            tools: Whether to enable tools for this invocation

        Returns:
            dict: Response dictionary with content, messages, and full responses from all iterations
        """
        current_messages = messages.copy()
        iterations = 0
        all_responses = []

        # Agent loop - continues until model stops calling tools
        while iterations < self.max_iterations:
            iterations += 1
            self._current_iteration = iterations  # Track for tool usage

            # Build API call parameters
            call_params = {
                "model": self.model,
                "messages": current_messages,
                "temperature": self.temperature
            }

            # determines whether tools are offered or not
            if tools:
                call_params["tools"] = self.tools
                call_params["tool_choice"] = "auto"


            #print(call_params)
            
            try:
                response = self.openai_client.chat.completions.create(**call_params)
                response_dict = response.model_dump() if response else None

                if response_dict is None:
                    return {
                        "content": "Error: Received None response from LLM",
                        "messages": current_messages,
                        "all_responses": all_responses,
                        "iterations": iterations,
                        "error": "None response"
                    }

                all_responses.append(response_dict)
                self._update_metadata(response_dict)
            except Exception as e:
                return {
                    "content": f"Error calling LLM: {str(e)}",
                    "messages": current_messages,
                    "all_responses": all_responses,
                    "iterations": iterations,
                    "error": str(e)
                }
            # OpenAI response format  example
            # {
            #   "id": "chatcmpl-abc123",
            #   "object": "chat.completion",
            #   "created": 1699896916,
            #   "model": "gpt-4o-mini",
            #   "choices": [
            #     {
            #       "index": 0,
            #       "message": {
            #         "role": "assistant",
            #         "content": null,
            #         "tool_calls": [
            #           {
            #             "id": "call_abc123",
            #             "type": "function",
            #             "function": {
            #               "name": "get_current_weather",
            #               "arguments": "{\n\"location\": \"Boston, MA\"\n}"
            #             }
            #           }
            #         ]
            #       },
            #       "logprobs": null,
            #       "finish_reason": "tool_calls"
            #     }
            #   ],
            #   "usage": {
            #     "prompt_tokens": 82,
            #     "completion_tokens": 17,
            #     "total_tokens": 99,
            #     "completion_tokens_details": {
            #       "reasoning_tokens": 0,
            #       "accepted_prediction_tokens": 0,
            #       "rejected_prediction_tokens": 0
            #     }
            #   }
            # }
            if "choices" not in response_dict or not response_dict["choices"]:

                return {
                    "content": "Error: No choices in response",
                    "messages": current_messages,
                    "all_responses": all_responses,
                    "iterations": iterations,
                    "error": "Invalid response structure",
                    "response_dict": response_dict
                }

            assistant_message = response_dict["choices"][0]["message"]
            current_messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls")

            if not tool_calls:
                return {
                    "content": assistant_message.get("content"),
                    "messages": current_messages,
                    "all_responses": all_responses,
                    "iterations": iterations
                }

            for tool_call in tool_calls:
                print(f"Using tool {tool_call}")
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])

                tool_result = self._execute_tool_call(function_name, function_args)

                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": function_name,
                    "content": tool_result
                })

        return {
            "content": "Max iterations reached",
            "messages": current_messages,
            "all_responses": all_responses,
            "iterations": iterations,
            "warning": f"Agent reached maximum iterations ({self.max_iterations})"
        }

    def cleanup(self) -> None:
        """
        Cleanup agent resources.

        Closes browser manager if browser tools are loaded.
        Safe to call multiple times.
        """
        if self.browser_manager is not None:
            try:
                self.browser_manager.close()
            except Exception as e:
                print(f"[BaseAgent] Error during cleanup: {e}")

    # Context manager protocol
    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup resources."""
        self.cleanup()
        return False  # Don't suppress exceptions

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup


    @abstractmethod
    def run(self,prompt):
        pass


    