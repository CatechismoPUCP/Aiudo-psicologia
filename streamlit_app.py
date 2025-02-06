import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Constants for model configuration
MODEL_NAME = "gemini-1.5-flash"
SYSTEM_PROMPT_FILE = "system_prompt.txt"
GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
SAFETY_SETTINGS = {  # Disable safety filters (adjust as needed)
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


def main():
    st.title("Gemini Chatbot")

    # API Key input
    api_key = st.text_input("Enter your Gemini API Key:", type="password")

    if api_key:
        # Initialize Gemini API (only once)
        genai.configure(api_key=api_key)

        try:
            # Load system prompt (only once)
            with open(SYSTEM_PROMPT_FILE, "r") as f:
                system_prompt = f.read()

            # Initialize model and chat session (only once)
            model = genai.GenerativeModel(
                model_name=MODEL_NAME,
                generation_config=GENERATION_CONFIG,
                safety_settings=SAFETY_SETTINGS,
            )
            chat_session = model.start_chat(context=system_prompt)

            # Initialize chat history in session state
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat history
            for message in st.session_state.messages:
                st.write(f"User: {message['user']}")
                st.write(f"Bot: {message['bot']}")

            # User input
            user_input = st.text_input("Enter your message:")

            if user_input:
                # Send message and get response
                response = chat_session.send_message(user_input)

                # Update chat history
                st.session_state.messages.append({"user": user_input, "bot": response.text})

                # Display bot's response
                st.write(f"Bot: {response.text}")

                # Rerun Streamlit to update the display
                st.experimental_rerun()

        except FileNotFoundError:
            st.error(f"System prompt file not found: {SYSTEM_PROMPT_FILE}")
        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
