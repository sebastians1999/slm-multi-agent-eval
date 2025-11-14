from dotenv import load_dotenv
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_interpreter import execute_python


if __name__ == "__main__":

    load_dotenv()

    test_code = """
print("Hello from E2B sandbox!")
result = 2 + 2
print(f"2 + 2 = {result}")
result
"""

    print("Testing execute_python with simple calculation...")
    response = execute_python(code=test_code)

    print("\n=== Response ===")
    print(response)
