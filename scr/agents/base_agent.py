from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from datetime import datetime
import uuid


class StructuredOutput(BaseModel):
   
    model_answer: Optional[str] 
    reasoning_trace: Optional[str] 


class AgentMetaData(BaseModel):
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    model_name: str
    temperature: float
    base_url: Optional[str] = None
    iterations: int = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    total_energy_joules: float = 0.0
    total_duration_seconds: float = 0.0
    average_watts: float = 0.0


class BaseMultiAgent(ABC):

    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        **kwargs
    ):
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.api_key = api_key
        self.config = kwargs

        self.openai_client = self._create_openai_client()



    def _create_openai_client(self) -> OpenAI:
        """Create and return the raw OpenAI client."""
        return OpenAI(api_key=self.api_key,base_url=self.base_url)



    def invoke(self, messages):
        """
        Invoke the LLM with messages and return the response as a dictionary.

        Args:
            messages: List of message dicts, e.g., [{"role": "user", "content": "..."}]

        Returns:
            dict: Response dictionary with choices, usage, and energy_consumption (if available)
        """
        
        
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        
        
        return response.model_dump()


    
    @abstractmethod
    def run(self):
        pass
