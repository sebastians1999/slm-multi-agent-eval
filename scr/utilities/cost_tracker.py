"""
Cost tracking utility for LLM API calls using Artificial Analysis pricing data.
"""

import requests
from typing import Dict, Any, Optional, Tuple
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class CostTracker:
    """
    Tracks and calculates costs for LLM API calls using Artificial Analysis pricing data.
    """

    def __init__(self):
        """
        Initialize the CostTracker.

        API key is loaded from ARTIFICIAL_ANALYSIS_API_KEY environment variable.
        """
        self.api_key = os.getenv("ARTIFICIAL_ANALYSIS_API_KEY")
        if not self.api_key:
            logger.warning("No Artificial Analysis API key found in environment. Cost tracking will be disabled.")
        self.base_url = "https://artificialanalysis.ai/api/v2/data/llms/models"
        self._cache: Dict[str, Dict[str, Any]] = {}

    def fetch_model_pricing(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch pricing data for a specific model from Artificial Analysis API.

        Args:
            model_name: Name or slug of the model

        Returns:
            Dictionary containing pricing and model information, or None if not found
        """
        if not self.api_key:
            logger.warning("Cannot fetch pricing data: No API key available")
            return None

        # Check cache first
        if model_name in self._cache:
            return self._cache[model_name]

        try:
            headers = {"x-api-key": self.api_key}
            response = requests.get(self.base_url, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get("status") != 200:
                logger.error(f"API returned non-200 status: {data.get('status')}")
                return None

            # Search for the model by name or slug
            models = data.get("data", [])
            for model in models:
                if model.get("name") == model_name or model.get("slug") == model_name:
                    # Cache the result
                    self._cache[model_name] = model
                    return model

            logger.warning(f"Model '{model_name}' not found in Artificial Analysis data")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching pricing data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    def calculate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate the total cost for a model inference based on token usage.

        Args:
            model_name: Name or slug of the model
            input_tokens: Number of input/prompt tokens used
            output_tokens: Number of output/completion tokens used

        Returns:
            Dictionary containing cost breakdown or None if pricing not available:
            {
                "input_cost": float,
                "output_cost": float,
                "total_cost": float,
                "currency": "USD",
                "input_tokens": int,
                "output_tokens": int,
                "model_name": str
            }
        """
        model_data = self.fetch_model_pricing(model_name)

        if not model_data:
            return None

        pricing = model_data.get("pricing")
        if not pricing:
            logger.warning(f"No pricing data available for model '{model_name}'")
            return None

        price_per_1m_input = pricing.get("price_1m_input_tokens")
        price_per_1m_output = pricing.get("price_1m_output_tokens")

        if price_per_1m_input is None or price_per_1m_output is None:
            logger.warning(f"Incomplete pricing data for model '{model_name}'")
            return None

        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * price_per_1m_input
        output_cost = (output_tokens / 1_000_000) * price_per_1m_output
        total_cost = input_cost + output_cost

        return {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "currency": "USD",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model_name": model_name,
            "price_per_1m_input_tokens": price_per_1m_input,
            "price_per_1m_output_tokens": price_per_1m_output,
        }


def calculate_inference_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to calculate cost for a single inference.

    API key is loaded from ARTIFICIAL_ANALYSIS_API_KEY environment variable.

    Args:
        model_name: Name or slug of the model
        input_tokens: Number of input/prompt tokens used
        output_tokens: Number of output/completion tokens used

    Returns:
        Dictionary containing cost breakdown or None if pricing not available
    """
    tracker = CostTracker()
    return tracker.calculate_cost(
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )
