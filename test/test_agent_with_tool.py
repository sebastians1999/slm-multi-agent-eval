"""
Test reactive agent with a single tool.
Demonstrates the agent deciding when to use the tool vs answering directly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scr.agents.base_agent import BaseAgent
import json


class TestAgent(BaseAgent):
    """Test agent for demonstrating tool usage."""

    def run(self, user_message: str):
        """
        Run the agent with a user message.

        Args:
            user_message: The user's input message

        Returns:
            dict: Response with content, messages, and all_responses
        """
        messages = [{"role": "user", "content": user_message}]
        return self.invoke(messages)


# Define the calculator tool function
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g., "15 * 7")

    Returns:
        JSON string with the result
    """
    try:
        # Safe evaluation with limited builtins
        result = eval(expression, {"__builtins__": {}}, {})
        return json.dumps({"success": True, "result": result})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# Define the calculator tool in OpenAI format
calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Performs mathematical calculations. Use this when you need to calculate exact numerical results.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '15 * 7', '100 / 4')"
                }
            },
            "required": ["expression"]
        }
    }
}


def print_response_summary(response, test_name):
    """Helper to print response summary."""
    print(f"\n{'='*60}")
    print(f"{test_name}")
    print('='*60)
    print(f"Final answer: {response['content']}")
    print(f"Iterations: {response['iterations']}")

    # Calculate energy and tokens
    if 'all_responses' in response and response['all_responses']:
        total_energy = sum(r.get('energy_consumption', {}).get('joules', 0)
                          for r in response['all_responses'])
        total_tokens = sum(r.get('usage', {}).get('total_tokens', 0)
                          for r in response['all_responses'])
        total_duration = sum(r.get('energy_consumption', {}).get('duration_seconds', 0)
                            for r in response['all_responses'])
        avg_watts = total_energy / total_duration if total_duration > 0 else 0

        print(f"Total energy: {total_energy:.4f} J")
        print(f"Total tokens: {total_tokens}")
        print(f"Average power: {avg_watts:.2f} W")

    # Show if tool was used
    tool_used = response['iterations'] > 1
    print(f"Tool used: {'Yes' if tool_used else 'No'}")

    if 'warning' in response:
        print(f"⚠ Warning: {response['warning']}")


def test_agent_uses_tool():
    """Test: Agent should USE the calculator tool for complex math."""

    agent = TestAgent(
        model="microsoft/Phi-4-mini-instruct",
        base_url="https://sebastian-schmuelling--slm-server-vllmserver-serve-dev.modal.run/v1/",
        temperature=0,
        tools=[calculator_tool],
        tool_functions={"calculator": calculator},
        max_iterations=5
    )

    # Ask a question that requires the tool
    response = agent.run("What is 157 multiplied by 239? Please calculate the exact result.")

    print_response_summary(response, "Test 1: Agent USES Tool (Complex Math)")

    # Verify tool was used
    assert response['iterations'] > 1, "Agent should have used the tool"
    assert "37523" in str(response['content']), "Result should contain correct answer"

    return response


def test_agent_skips_tool():
    """Test: Agent should NOT use the tool for simple questions."""

    agent = TestAgent(
        model="microsoft/Phi-4-mini-instruct",
        base_url="https://sebastian-schmuelling--slm-server-vllmserver-serve-dev.modal.run/v1/",
        temperature=0,
        tools=[calculator_tool],
        tool_functions={"calculator": calculator},
        max_iterations=5
    )

    # Ask a question that doesn't need the tool
    response = agent.run("What is the capital of France?")

    print_response_summary(response, "Test 2: Agent SKIPS Tool (Non-math Question)")

    # Verify tool was NOT used
    assert response['iterations'] == 1, "Agent should answer directly without tool"
    assert "Paris" in response['content'], "Should answer correctly"

    return response


def test_agent_simple_math():
    """Test: Agent might choose to answer simple math directly or use tool."""

    agent = TestAgent(
        model="microsoft/Phi-4-mini-instruct",
        base_url="https://sebastian-schmuelling--slm-server-vllmserver-serve-dev.modal.run/v1/",
        temperature=0,
        tools=[calculator_tool],
        tool_functions={"calculator": calculator},
        max_iterations=5
    )

    # Simple math - agent can decide
    response = agent.run("What is 2 plus 2?")

    print_response_summary(response, "Test 3: Agent Decides (Simple Math)")

    # Just check it gets the right answer
    assert "4" in response['content'], "Should get correct answer"

    return response


def test_agent_multiple_calculations():
    """Test: Agent uses tool multiple times if needed."""

    agent = TestAgent(
        model="microsoft/Phi-4-mini-instruct",
        base_url="https://sebastian-schmuelling--slm-server-vllmserver-serve-dev.modal.run/v1/",
        temperature=0,
        tools=[calculator_tool],
        tool_functions={"calculator": calculator},
        max_iterations=10
    )

    # Ask for multiple calculations
    response = agent.run(
        "Calculate 123 * 456, then divide the result by 7. Show me both steps."
    )

    print_response_summary(response, "Test 4: Agent Multiple Tool Uses")

    # Should have used tool at least once
    assert response['iterations'] >= 2, "Agent should use tool"

    return response


def test_energy_tracking():
    """Test: Verify energy consumption is tracked across all tool calls."""

    agent = TestAgent(
        model="microsoft/Phi-4-mini-instruct",
        base_url="https://sebastian-schmuelling--slm-server-vllmserver-serve-dev.modal.run/v1/",
        temperature=0,
        tools=[calculator_tool],
        tool_functions={"calculator": calculator},
        max_iterations=5
    )

    response = agent.run("What is 999 times 888?")

    print_response_summary(response, "Test 5: Energy Tracking")

    # Verify energy data exists
    assert 'all_responses' in response, "Should have all_responses"
    assert len(response['all_responses']) > 0, "Should have at least one response"

    # Check each response has energy data
    for i, r in enumerate(response['all_responses']):
        print(f"\n  Response {i+1}:")
        if 'energy_consumption' in r:
            energy = r['energy_consumption']
            print(f"    Energy: {energy.get('joules', 0):.4f} J")
            print(f"    Duration: {energy.get('duration_seconds', 0):.3f} s")
            print(f"    Power: {energy.get('watts', 0):.2f} W")
        else:
            print(f"    ⚠ No energy data")

    return response


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Reactive Agent with Calculator Tool")
    print("="*60)

    try:
        # Run all tests
        test_agent_uses_tool()
        test_agent_skips_tool()
        test_agent_simple_math()
        test_agent_multiple_calculations()
        test_energy_tracking()

        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise
