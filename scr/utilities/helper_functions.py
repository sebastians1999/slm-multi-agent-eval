import re 


def extract_final_answer(message: str) -> str|None:
    
    if not message:
        return None 
    
    pattern = r'FINAL ANSWER:\s*(.+?)(?:\n|$)'   
    
    match = re.search(pattern,message, re.IGNORECASE | re.DOTALL)
    
    if match:
        answer = match.group(1).strip()
        return answer
    
    return None