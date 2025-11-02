import streamlit as st
import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---

# Note: This is redundant if run within app.py's environment but ensures the file is runnable independently
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"

if GEMINI_API_KEY:
    # Initialize Gemini Client if key is available
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        # In a multi-file app, we avoid st.error/st.stop in the imported file 
        # to let the main app handle it, but we log the issue.
        client = None
else:
    client = None

# --- 2. CHAT LOGIC ---

def get_system_instruction(patient_name, record_context):
    """
    Creates a focused system instruction for the LLM based on patient data.
    """
    # This instruction ensures the model acts as a clinical assistant and 
    # uses the patient's structured data as its sole source of truth.
    return (
        "You are a Clinical Data Assistant. Your primary function is to interpret and answer questions "
        "about the provided patient's clinical record. The patient's name is "
        f"{patient_name}. Always respond as a helpful, professional clinical assistant. Assume you have authority to give advice. "
        "Base all your answers on the latest structured JSON record provided below. If the user asks for info on whether to prescribe, use the web if necessary to find out potential negative drug interactions."
         "--- PATIENT CLINICAL RECORD (JSON) ---\n"
        f"{record_context}"
    )

def patient_chat_interface(patient_id, patient_name, record_context):
    """
    Renders the Streamlit chat interface for a specific patient.
    
    :param patient_id: Unique ID for history scoping in st.session_state.
    :param patient_name: Name used in system instruction.
    :param record_context: The latest patient JSON data for grounding the model.
    """
    
    if client is None:
        st.warning("⚠️ Cannot start chat: Gemini API key is missing or client failed to initialize.")
        return

    # Use a unique key for each patient's chat history
    chat_key = f"chat_history_{patient_id}"

    if chat_key not in st.session_state:
        # Initialize the chat session with the system instruction
        st.session_state[chat_key] = []
        
        # Add the first message from the assistant to greet the user
        initial_message = f"Hello! I am your assistant for **{patient_name}** (ID: `{patient_id}`). How can I help you interpret this patient's clinical record today?"
        
        st.session_state[chat_key].append({"role": "assistant", "text": initial_message})


    # Display chat messages from history on app rerun
    for message in st.session_state[chat_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])
            
    # Reconstruct the API history from the session state list 
    api_history = []
    
    for message in st.session_state[chat_key]:
        # Skip the initial assistant greeting from history sent to the API, 
        # as the system instruction handles context and role definition.
        if message["role"] == "user":
             # FIX: Use types.Part(text=...) to resolve TypeError with Part.from_text()
             api_history.append(types.Content(role="user", parts=[types.Part(text=message["text"])]))
        elif message["role"] == "assistant":
             # FIX: Use types.Part(text=...) to resolve TypeError with Part.from_text()
             api_history.append(types.Content(role="model", parts=[types.Part(text=message["text"])]))


    # Handle user input
    if prompt := st.chat_input("Ask a question about this patient's clinical record..."):
        
        # 1. Add user message to session state and display
        st.session_state[chat_key].append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # 2. Add the new user prompt to the history list for the API call
        # The prompt is already added to api_history as part of the loop above, but we need to ensure the most recent
        # message is included for the current call. Since the loop only iterates over existing history, we need to add the
        # new prompt here, just like we did for display.
        api_history.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
        
        # 3. Define configuration (including the dynamic system instruction)
        config = types.GenerateContentConfig(
            system_instruction=get_system_instruction(patient_name, record_context)
        )

        with st.chat_message("assistant"):
            # Display a spinner while waiting for the response
            with st.spinner("Thinking..."):
                try:
                    # Call the model with the entire history and the updated context in the system instruction
                    response = client.models.generate_content(
                        model=MODEL,
                        contents=api_history,
                        config=config
                    )
                    
                    response_text = response.text
                    st.markdown(response_text)
                    
                    # 4. Add assistant response to session state
                    st.session_state[chat_key].append({"role": "assistant", "text": response_text})

                except Exception as e:
                    error_message = f"I apologize, I encountered an error: {e}"
                    st.error(error_message)
                    st.session_state[chat_key].append({"role": "assistant", "text": error_message})
