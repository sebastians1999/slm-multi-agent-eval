#!/usr/bin/env python3
"""
Test script for the Python interpreter tool
"""
from scr.agents.tools.python_interpreter import execute_python

def test_simple_calculation():
    """Test basic calculation"""
    print("=" * 60)
    print("Test 1: Simple calculation")
    print("=" * 60)

    code = """
result = 2 + 2
print(f"2 + 2 = {result}")
result
"""
    result = execute_python(code=code)
    print(f"Result: {result}")
    print(f"Result type: {type(result)}")
    print()


def test_data_analysis():
    """Test data analysis with pandas"""
    print("=" * 60)
    print("Test 2: Data analysis")
    print("=" * 60)

    code = """
import pandas as pd

data = {'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35]}
df = pd.DataFrame(data)
print(df)
print(f"\\nAverage age: {df['age'].mean()}")
"""
    result = execute_python(code=code)
    print(f"Result: {result}")
    print()


def test_web_scraping():
    """Test the exact code the agent tried to use"""
    print("=" * 60)
    print("Test 3: Web scraping (agent's code)")
    print("=" * 60)

    code = """
from urllib.request import urlopen
from bs4 import BeautifulSoup

# URL of the paper submitted to arXiv.org in June 2022
# Note: This ID actually exists (Real Quadratic Fields paper)
paper_url_2022 = 'https://arxiv.org/abs/2206.12345' 

# URL of the Physics and Society article submitted to arXiv.org on August 11, 2016
# Note: This ID actually exists (Demixing Sparse Signals paper)
paper_url_2016 = 'https://arxiv.org/abs/1608.01234'

# Fetch the content of the 2022 paper
try:
    response_2022 = urlopen(paper_url_2022)
    soup_2022 = BeautifulSoup(response_2022, 'html.parser')
    # Extract figure details (hypothetical)
    figure_labels_2022 = []
    for img in soup_2022.find_all('img'):
        if 'figure' in img.get('alt', '') or 'graph' in img.get('alt', ''):
            figure_labels_2022 = [label.strip() for label in img.get('alt', '').split(',') if label.strip()]
    print('Labels from 2022 paper figure:', figure_labels_2022)

    # Fetch the content of the 2016 paper
    response_2016 = urlopen(paper_url_2016)
    soup_2016 = BeautifulSoup(response_2016, 'html.parser')
    # Extract relevant text from the 2016 paper
    text_2016 = soup_2016.get_text()
    # Look for terms describing a type of society
    society_terms = [word for word in text_2016.split() if word.lower() in ['society', 'societal', 'social', 'community', 'civil', 'democratic', 'utopian', 'post-scarcity']]
    print('Society-related terms in 2016 paper:', society_terms)

except Exception as e:
    print('Error fetching or parsing the papers:', e)
"""
    result = execute_python(code=code)
    print(f"Result: {result}")
    print(f"Result type: {type(result)}")
    print()


def test_simple_web_request():
    """Test simple web request"""
    print("=" * 60)
    print("Test 4: Simple web request")
    print("=" * 60)

    code = """
from urllib.request import urlopen

try:
    response = urlopen('https://httpbin.org/get')
    content = response.read().decode('utf-8')
    print("Successfully fetched URL")
    print(f"Response length: {len(content)} characters")
    print(f"First 200 chars: {content[:200]}")
except Exception as e:
    print(f"Error: {e}")
"""
    result = execute_python(code=code)
    print(f"Result: {result}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING PYTHON INTERPRETER TOOL")
    print("=" * 60 + "\n")

    try:
        test_simple_calculation()
        test_data_analysis()
        test_simple_web_request()
        test_web_scraping()
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("TESTS COMPLETED")
    print("=" * 60)
