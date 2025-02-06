import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Function to read the system prompt from a file
def read_system_prompt(file_path):
    with open(file_path, "r") as file:
        return file.read()

# Streamlit app
def main():
    st.title("Gemini Chatbot")

    # Input for API key
    api_key = st.text_input("Enter your Gemini API Key:", type="password")

    if api_key:
        # Configure the API key
        genai.configure(api_key=api_key)

        # Read the system prompt from a file
        system_prompt = read_system_prompt("system_prompt.txt")

        # Create the model
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        # Define safety settings to disable or adjust filters
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings,  # Apply custom safety settings
        )

        # Start a chat session with the system prompt
        chat_session = model.start_chat(history=[])

        # Input for user message
        user_input = st.text_input("Enter your message:")

        if user_input:
            # Send the user's message to the model
            response = chat_session.send_message(system_prompt + "\n" + user_input)

            # Display the response
            st.write("Response:")
            st.write(response.text)

if __name__ == "__main__":
    main()
