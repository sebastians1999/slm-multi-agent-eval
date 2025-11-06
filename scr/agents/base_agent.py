from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from datetime import datetime
import uuid


class StructuredOutput(BaseModel):
   
    model_answer: Optional[str] 
    reasoning_trace: Optional[str] 


class AgentRunMetadata(BaseModel):
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



    def invoke(self,prompt: str,session_id: str,metadata_tracker: Dict[str, AgentRunMetadata]):
  
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )

        # Debug: Inspect response structure
        # print("\n[DEBUG] Response inspection:")
        # print(f"  Response type: {type(response)}")

        # response_dict = response.model_dump()
        # print(f"  Available keys: {list(response_dict.keys())}")


        # if 'energy_consumption' in response_dict:
        #     print(f"  Energy data found: {response_dict['energy_consumption']}")
        # else:
        #     print("  No energy_consumption field found")

        # message = response.choices[0].message.content
        
        # energy_meta_data = response_dict.get('energy_consumption', 'Not found')

        # return message, energy_meta_data
        return response.model_dump()

    