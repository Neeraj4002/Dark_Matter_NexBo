import re

def parse_finance_input(input_text: str) -> dict:
    print("DEBUG: Input text:", input_text)
    budget = None
    currency = None
    sector = None

    # Extract budget using regex.
    budget_pattern = re.compile(
        r"(\d+(?:[,.]\d+)?)(?:\s*(lakh|million))?\s*(rupees|dollars)?",
        re.IGNORECASE
    )
    budget_match = budget_pattern.search(input_text)
    if budget_match:
        print("DEBUG: Budget match groups:", budget_match.groups())
        num_str = budget_match.group(1).replace(',', '')
        try:
            num = float(num_str)
        except ValueError:
            num = None

        multiplier = 1
        unit = budget_match.group(2)
        if unit:
            unit = unit.lower()
            if unit == "lakh":
                multiplier = 100000
            elif unit == "million":
                multiplier = 1000000

        if num is not None:
            budget = num * multiplier

        currency_text = budget_match.group(3)
        if currency_text:
            currency_text = currency_text.lower()
            if "rupees" in currency_text:
                currency = "INR"
            elif "dollars" in currency_text:
                currency = "USD"
    else:
        print("DEBUG: No budget match found.")

    # Extract sector by searching through a predefined list of sectors.
    sectors = [
        "Technology", "Healthcare", "Financial Services", "Energy",
        "Consumer Goods", "Utilities", "Real Estate", "Industrial"
    ]
    for s in sectors:
        print(f"DEBUG: Checking for sector '{s}' in input...")
        match = re.search(r"\b" + re.escape(s) + r"\b", input_text, re.IGNORECASE)
        if match:
            print("DEBUG: Found sector match:", match.group(0))
            sector = s
            break

    if not sector:
        print("DEBUG: No sector match found.")

    result = {
        "budget": budget,
        "currency": currency,
        "criteria": {"sector": sector} if sector else {}
    }
    print("DEBUG: Parse result:", result)
    return result

# For standalone testing
if __name__ == "__main__":
    test_input = "Recommend few stocks for my 3 lakh rupees budget and Technology sector as criteria"
    parsed = parse_finance_input(test_input)
    print("Final parsed output:", parsed)
