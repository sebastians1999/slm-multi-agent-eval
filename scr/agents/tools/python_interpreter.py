from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("E2B_API_KEY")
if not api_key:
    raise ValueError("E2B_API_KEY environment variable is not set")

# OpenAI-compatible tool schema
PYTHON_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Execute Python code in a secure Jupyter notebook cell and return the result. Use this to perform calculations, data analysis, or run Python scripts.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute in a single cell"
                }
            },
            "required": ["code"],
            "additionalProperties": False
        }
    }
}

def execute_python(**kwargs):
    """
    Execute Python code in a secure E2B sandbox.

    Args:
        code (str): Python code to execute

    Returns:
        str: Execution result text output
    """
    
    with Sandbox.create(api_key=api_key) as sandbox:
        code = kwargs['code']
        execution = sandbox.run_code(code)

        if execution.error:
            return f"Error: {execution.error.name}: {execution.error.value}\n{execution.error.traceback}"
            
        output_logs = "\n".join(execution.logs.stdout)
        

        result = output_logs
        if execution.text:
            result += f"\nResult: {execution.text}"
            
        print("Python tool used.")
        
    return result

    