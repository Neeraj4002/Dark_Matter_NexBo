import json
import re
from typing import Optional, Union

def extract_json_from_response(response: Union[str, list]) -> Optional[dict]:
    """
    Extracts and parses JSON from an AI response that may be a string or list of strings.
    Handles cases where JSON is within markdown code blocks or plain text.
    
    Returns:
        dict if JSON is successfully extracted and parsed, else None.
    """
    # Step 1: Normalize input to a single string
    if isinstance(response, list):
        response = "\n".join(response)
    elif not isinstance(response, str):
        return None  # Invalid format

    # Step 2: Look for JSON code blocks first
    json_block_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1))
        except json.JSONDecodeError:
            pass  # Continue trying other methods

    # Step 3: Try to extract any JSON-looking object from the text
    json_like_match = re.search(r"(\{.*\})", response, re.DOTALL)
    if json_like_match:
        try:
            return json.loads(json_like_match.group(1))
        except json.JSONDecodeError:
            pass

    # Step 4: Try to directly parse the entire response
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None
