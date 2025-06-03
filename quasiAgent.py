import os
import re
from google import genai
from typing import List, Dict

# The Gemini API key should be set as an environment variable outside this script for security.
# Example (PowerShell): $env:GOOGLE_API_KEY="your-actual-gemini-key"

api_key = os.environ.get('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it before running the script.")
client = genai.Client(api_key=api_key)

def extract_code(text):
    """Extracts Python code from LLM response."""
    code_blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
    return code_blocks[0].strip() if code_blocks else text

def generate_response(messages: List[Dict]) -> str:
    """Call Gemini LLM directly using google-genai (new API)"""
    history = []
    for message in messages:
        role = 'user' if message['role'] in ['system', 'user'] else 'model'
        history.append({'role': role, 'parts': [message['content']]})
    chat = client.chats.create(model='gemini-2.5-pro-preview-05-06', history=history[:-1])
    response = chat.send_message(history[-1]['parts'][0])
    return response.text

def prompt_llm(prompt, context=None):
    """Send prompt to LLM, optionally with context."""
    messages = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": prompt})
    return generate_response(messages)

def main():
    # Step 1: Ask user for function description
    user_desc = input("Describe the Python function you want to create:\n")

    # Step 2: Ask LLM to write the function
    prompt1 = f"Write a basic Python function based on this description: {user_desc}"
    step1_response = prompt_llm(prompt1)
    print("\n--- Step 1: Basic Function ---\n")
    print(step1_response)
    code = extract_code(step1_response)

    # Step 3: Ask LLM to add documentation
    prompt2 = (
        "Add comprehensive documentation to this function, including:\n"
        "- Function description\n"
        "- Parameter descriptions\n"
        "- Return value description\n"
        "- Example usage\n"
        "- Edge cases\n"
        f"\nHere is the code:\n```python\n{code}\n```"
    )
    step2_response = prompt_llm(prompt2)
    print("\n--- Step 2: Documented Function ---\n")
    print(step2_response)
    documented_code = extract_code(step2_response)

    # Step 4: Ask LLM to add unit tests
    prompt3 = (
        "Add test cases using Python's unittest framework for the following code. "
        "Tests should cover basic functionality, edge cases, error cases, and various input scenarios.\n"
        f"\nHere is the documented code:\n```python\n{documented_code}\n```"
    )
    step3_response = prompt_llm(prompt3)
    print("\n--- Step 3: Function with Tests ---\n")
    print(step3_response)
    final_code = extract_code(step3_response)

    # Step 5: Save final code to file
    filename = input("\nEnter filename to save the final code (e.g., my_function.py): ")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_code)
    print(f"\nFinal code saved to {filename}")

if __name__ == "__main__":
    main()
