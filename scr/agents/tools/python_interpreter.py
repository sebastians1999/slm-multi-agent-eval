from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("E2B_API_KEY")
if not api_key:
    raise ValueError("E2B_API_KEY environment variable is not set")

def execute_python(**kwargs):
      with Sandbox.create(api_key=api_key) as sandbox:
          code = kwargs['code']
          execution = sandbox.run_code(code)
          result = execution.text
      return result
    