from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Callable
from pydantic import BaseModel
from openai import OpenAI
from .tools_list import tools
from .tool_functions import tool_functions
import json
    

class agent_meta_data(BaseModel):
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    total_energy_joules: float = 0.0
    total_duration_seconds: float = 0.0
    average_watts: float = 0.0


class BaseAgent(ABC):

    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        tools: Optional[List[Dict]] = tools,
        tool_functions: Optional[Dict[str, Callable]] = tool_functions,
        max_iterations: int = 10,
        **kwargs
    ):
        """
        Initialize the agent.
        """
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.api_key = api_key
        self.config = kwargs
        self.tools = tools or []
        self.tool_functions = tool_functions or {}
        self.max_iterations = max_iterations
        self.meta_data = agent_meta_data()

        self.openai_client = self._create_openai_client()

    
    
    

    def _create_openai_client(self) -> OpenAI:
        """Create and return the raw OpenAI client."""
        return OpenAI(api_key=self.api_key, base_url=self.base_url)


    def _update_metadata(self, response_dict: Dict) -> None:
        """Update metadata based on API response."""
        usage = response_dict.get("usage", {})

        self.meta_data.completion_tokens += usage.get("completion_tokens", 0)
        self.meta_data.prompt_tokens += usage.get("prompt_tokens", 0)
        self.meta_data.total_tokens += usage.get("total_tokens", 0)

        energy_consumption = response_dict.get("energy_consumption", {})
        self.meta_data.total_energy_joules += energy_consumption.get("joules", 0.0)
        self.meta_data.total_duration_seconds += energy_consumption.get("duration_seconds", 0.0)

        if self.meta_data.total_duration_seconds > 0:
            self.meta_data.average_watts = (
                self.meta_data.total_energy_joules / self.meta_data.total_duration_seconds
            )


    def _execute_tool_call(self, tool_name: str, tool_args: Dict) -> str:
        """Execute a tool function and return the result."""
        if tool_name not in self.tool_functions:
            return f"Error: Tool '{tool_name}' not found in tool_functions"

        try:
            tool_func = self.tool_functions[tool_name]
            result = tool_func(**tool_args)
            print(result)
            return str(result)
        except Exception as e:
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


            print(call_params)
            
            try:
                response = self.openai_client.chat.completions.create(**call_params)
                response_dict = response.model_dump()
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

    
    @abstractmethod
    def run(self,prompt):
        pass


    