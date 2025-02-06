import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Constants
MODEL_NAME = "gemini-1.5-flash"
SYSTEM_PROMPT_FILE = "system_prompt.txt"
GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

def main():
    st.title("Gemini Chatbot")

    api_key = st.text_input("Enter your Gemini API Key:", type="password")

    if api_key:
        genai.configure(api_key=api_key)

        try:
            with open(SYSTEM_PROMPT_FILE, "r") as f:
                system_prompt = f.read()

            model = genai.GenerativeModel(
                model_name=MODEL_NAME,
                generation_config=GENERATION_CONFIG,
                safety_settings=SAFETY_SETTINGS,
            )

            # Initialize chat session
            chat_session = model.start_chat()

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = [(None, system_prompt)]
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                st.write(f"User: {message['user']}")
                st.write(f"Bot: {message['bot']}")

            user_input = st.text_input("Enter your message:")

            if user_input:
                response = chat_session.send_message(
                    user_input, history=st.session_state.chat_history
                )

                st.session_state.chat_history.append((user_input, response.text))
                st.session_state.messages.append({"user": user_input, "bot": response.text})

                st.write(f"Bot: {response.text}")
                st.experimental_rerun()

        except FileNotFoundError:
            st.error(f"System prompt file not found: {SYSTEM_PROMPT_FILE}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
