import re
from typing import Optional


def extract_final_answer(text: Optional[str]) -> Optional[str]:
    """
    Extract the final answer from LLM response text.

    Looks for 'FINAL ANSWER:' marker and extracts everything after it.
    Cleans up the extracted answer by removing extra whitespace and quotes.

    Args:
        text: The full LLM response text

    Returns:
        The extracted final answer, or None if not found or text is None

    Example:
        >>> text = "Here is my reasoning...\\nFINAL ANSWER: egalitarian"
        >>> extract_final_answer(text)
        'egalitarian'
    """
    if not text:
        return None

    pattern = r'FINAL ANSWER:\s*(.+?)(?:\n|$)'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

    if not match:
        return None

    answer = match.group(1).strip()

    answer = re.sub(r'^["\']|["\']$', '', answer)

    return answer
