"""
Debug test to check if tool calling is working properly.
"""

import openai
import json

BASE_URL = "https://sebastian-schmuelling--slm-server-vllmserver-serve-dev.modal.run/v1/"
MODEL = "microsoft/Phi-4-mini-instruct"


def test_direct_openai_tool_call():
    """Test tool calling directly with OpenAI client to debug."""

    print("\n=== Direct OpenAI Client Tool Call Test ===\n")

    client = openai.OpenAI(
        api_key="EMPTY",
        base_url=BASE_URL
    )

    # Define the calculator tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Performs mathematical calculations. Use this for exact numerical results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

    # Make the request
    print("Making request with tools...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "What is 157 multiplied by 239? Use the calculator tool to get the exact result."}
        ],
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )

    print(f"\nResponse received!")
    response_dict = response.model_dump()

    # Print response structure
    print(f"\nResponse keys: {list(response_dict.keys())}")

    if 'choices' in response_dict and response_dict['choices']:
        choice = response_dict['choices'][0]
        message = choice['message']

        print(f"\nMessage keys: {list(message.keys())}")
        print(f"Finish reason: {choice.get('finish_reason')}")

        # Check for tool calls
        if 'tool_calls' in message and message['tool_calls']:
            print(f"\n✓ Model wants to use tool!")
            for tc in message['tool_calls']:
                print(f"  Tool: {tc['function']['name']}")
                print(f"  Arguments: {tc['function']['arguments']}")
        else:
            print(f"\n✗ Model did NOT call tool")
            if 'content' in message and message['content']:
                print(f"  Direct answer: {message['content']}")

    # Check for energy consumption
    if 'energy_consumption' in response_dict:
        energy = response_dict['energy_consumption']
        print(f"\nEnergy: {energy.get('joules', 0):.4f} J")

    print("\n" + "="*60)
    print("Full response (formatted):")
    print(json.dumps(response_dict, indent=2, default=str))

    return response_dict


def test_stronger_tool_prompt():
    """Test with a stronger prompt that really insists on using the tool."""

    print("\n\n=== Test with Stronger Tool Prompt ===\n")

    client = openai.OpenAI(
        api_key="EMPTY",
        base_url=BASE_URL
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Performs mathematical calculations. You MUST use this tool for any arithmetic operations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. When asked to perform calculations, you MUST use the calculator tool. Never calculate in your head."},
            {"role": "user", "content": "Calculate 157 * 239 using the calculator tool."}
        ],
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )

    response_dict = response.model_dump()

    if 'choices' in response_dict and response_dict['choices']:
        message = response_dict['choices'][0]['message']

        if 'tool_calls' in message and message['tool_calls']:
            print("✓ Model called tool with stronger prompt!")
        else:
            print("✗ Model still didn't call tool even with stronger prompt")
            print(f"Answer: {message.get('content', 'No content')}")

    return response_dict


def test_forced_tool_choice():
    """Test with tool_choice set to specific function."""

    print("\n\n=== Test with Forced Tool Choice ===\n")

    client = openai.OpenAI(
        api_key="EMPTY",
        base_url=BASE_URL
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Performs mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Calculate 157 * 239"}
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "calculator"}},
            temperature=0,
        )

        response_dict = response.model_dump()

        if 'choices' in response_dict and response_dict['choices']:
            message = response_dict['choices'][0]['message']

            if 'tool_calls' in message and message['tool_calls']:
                print("✓ Model called tool when forced!")
                print(f"  Arguments: {message['tool_calls'][0]['function']['arguments']}")
            else:
                print("✗ Model didn't call tool even when forced")

    except Exception as e:
        print(f"✗ Error with forced tool choice: {e}")

    return None


if __name__ == "__main__":
    print("="*60)
    print("Debugging Tool Calling")
    print("="*60)

    test_direct_openai_tool_call()
    test_stronger_tool_prompt()
    test_forced_tool_choice()

    print("\n" + "="*60)
    print("Debug tests completed")
    print("="*60)
