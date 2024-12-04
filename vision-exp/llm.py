import streamlit as st
import requests
import os
from dotenv import load_dotenv
import json
import re
from pydantic import BaseModel
from typing import List

# Load environment variables from .env file
load_dotenv('.env')

# Define a Pydantic model for the expected JSON response
class PossibleObjects(BaseModel):
    possible_objects: List[str]

def ask_text_query(
    text_prompt,
    model_name="gpt-4o-mini",
    api_base="https://api.openai.com/v1",
    timeout=10,
):
    """
    Sends a text prompt to the OpenAI API and returns the response.
    """
    try:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": text_prompt}],
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        }

        response = requests.post(
            f"{api_base}/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        # Extract the assistant's reply
        output = data["choices"][0]["message"]["content"]

    except Exception as e:
        output = f"Error: {str(e)}"

    return output

def postprocess_llm(response):
    """
    Extracts the JSON part from the assistant's response.
    """
    try:
        json_string = re.search(r"```json\n(.*?)\n```", response, re.DOTALL).group(1)
        return PossibleObjects(**json.loads(json_string))
    except AttributeError:
        raise ValueError("The response does not contain a valid JSON block.")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in the response.")

def main():
    st.title("🔍 Object Identifier")

    st.write(
        """
        Enter a command or query, and the system will identify possible objects or entities 
        that a robot should interact with based on the input.
        """
    )

    # Text input for the user's prompt
    prompt = st.text_input("Enter your command or query:")

    if st.button("Submit") and prompt:
        final_prompt = (
            f"""
Given is user query: "{prompt}".
We are currently in an indoor environment that can be a warehouse, office, or factory. 
Commands are given to a robot to navigate the environment.
Which objects or entities could the user be referring to when they say "{prompt}"? 
The robot would then need to go to that object or entity.
Remember that the robot should be able to go to the possible object and then perform an action suitable to the user query.
Return the possible objects in a JSON format.
"""
            + """
Eg. if the query is "go upstairs", the possible objects could be "stairs", "staircase", "steps". Hence the JSON output would be:
{
    "possible_objects": [
        "stairs",
        "staircase",
        "steps"
    ]
}
"""
        )

        with st.spinner("Processing..."):
            response = ask_text_query(final_prompt)
            try:
                objects = postprocess_llm(response)
                st.success("Possible Objects Identified:")
                st.json(objects.model_dump())
            except ValueError as ve:
                st.error(f"Failed to parse response: {ve}")
                st.text(response)

if __name__ == "__main__":
    main()
