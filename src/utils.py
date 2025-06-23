import re

def extract_answer(text_response: str, pattern: str = r"Answer:\s*(.*)") -> str:
    """
    Extracts the answer from the text response using a specified regex pattern.

    :param text_response: The text response from which to extract the answer.
    :param pattern: The regex pattern to use for extraction (default is r"Answer:\s*(.*)").
    :return: The extracted answer or the original text if no match is found.
    """
    match = re.search(pattern, text_response, re.DOTALL)
    if match:
        answer_text = match.group(1).strip()
        return answer_text
    else:
        return "answer not found in the response"