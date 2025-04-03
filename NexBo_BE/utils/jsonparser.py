import re
import json

def parse_ai_response(response_list):
    """
    Extracts the JSON block from the AI response list and parses it.
    
    Parameters:
        response_list (list): A list of strings returned by the AI.
    
    Returns:
        dict: A Python dictionary representing the parsed JSON object.
    
    Raises:
        ValueError: If no valid JSON block is found or JSON parsing fails.
    """
    json_block = None

    # Iterate over the list to find a string containing a JSON code block
    for part in response_list:
        # Look for a code block wrapped with ```json ... ```
        match = re.search(r"```json\s*(\{.*?\})\s*```", part, re.DOTALL)
        if match:
            json_block = match.group(1)
            break

    if not json_block:
        raise ValueError("No JSON block found in the AI response.")

    try:
        # Parse the JSON block
        parsed_json = json.loads(json_block)
    except json.JSONDecodeError as e:
        raise ValueError("Failed to parse JSON: " + str(e))

    return parsed_json
