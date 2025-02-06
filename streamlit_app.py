import streamlit as st
import google.generativeai as genai
import os

# --- Configuration and Error Handling ---

# Check for API Key (Crucial First Step)
if "GEMINI_API_KEY" not in os.environ:
    st.error("Error:  The GEMINI_API_KEY environment variable is not set.  Please set it before running the app.")
    st.stop()  # Stop execution if the key is missing

genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def load_system_prompt(filepath="system_prompt.txt"):
    """Loads the system prompt from a file, handling potential errors."""
    try:
        with open(filepath, "r") as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Error: System prompt file not found at '{filepath}'. Please create the file.")
        return ""  # Return an empty string so the program doesn't crash
    except Exception as e:
        st.error(f"An unexpected error occurred reading the system prompt: {e}")
        return ""


# --- Gemini Setup ---

def get_gemini_response(user_input, system_prompt):
    """Gets a response from the Gemini model, handling potential errors."""
    generation_config = {
        "temperature": 0.7,  #  Adjust as needed
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        }

    ]

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",  # Use a consistent model name.  "gemini-2.0-pro-exp-02-05" might be invalid
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    # Use the chat interface for a conversational flow (better than raw generation)
    chat = model.start_chat(history=[])

    # Send the system prompt as the *first* message in the history.
    if system_prompt:  # Only send if the prompt was loaded successfully
        chat.send_message(system_prompt)  # System prompt goes *first*


    try:
        response = chat.send_message(user_input)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while communicating with Gemini: {e}")
        return "Error: Could not get a response from Gemini."



# --- Streamlit App ---

def main():
    st.title("Gemini AI Chatbot with Custom System Prompt")

    system_prompt = load_system_prompt()  # Load the prompt

    user_input = st.text_area("Enter your message:", height=150)

    if st.button("Send"):
        if user_input:
            with st.spinner("Generating response..."):  # Show a spinner
                response = get_gemini_response(user_input, system_prompt)
            st.markdown("### Response:")
            st.write(response)  # Use st.write for displaying the response
        else:
            st.warning("Please enter a message.")



if __name__ == "__main__":
    main()
