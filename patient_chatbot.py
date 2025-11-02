# patient_chatbot.py

import streamlit as st
import json
from google import genai
from google.genai import types
from google.cloud import firestore
import os
from dotenv import load_dotenv
import pandas as pd
from typing import Dict, Any, List

# --- Setup & Caching (Relies on app.py for initial setup) ---

# NOTE: We rely on the client and db objects initialized in app.py
# The constants are re-defined here for clarity but are sourced externally.
load_dotenv()
MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Helper to get client initialized in app.py's session state ---
def get_client_from_session():
    # This assumes the client was initialized and cached correctly in app.py
    # We will fetch it from the external module scope if possible
    try:
        from app import client as app_client
        return app_client
    except (ImportError, AttributeError):
        # Fallback if running standalone or client hasn't been defined
        return genai.Client(api_key=GEMINI_API_KEY)


# --- Data Manipulation Helper (CRITICAL FOR UPDATES) ---

def update_record_data(record: Dict[str, Any], action: str, target: str, details: Dict[str, str]) -> Dict[str, Any]:
    """Applies the AI's requested modification (add/remove) to the patient's record lists."""
    
    # 1. Check if the target exists and is a list (problems or medications)
    if target not in record or not isinstance(record.get(target), list):
        record[target] = []
        
    target_list = record[target]
    
    # Ensure 'details' has the required keys for comparison
    required_keys = ['standard_name', 'standard_code_type', 'standard_code_value']
    if not all(k in details for k in required_keys):
        st.error(f"Invalid details for update: Missing required code fields.")
        return record

    if action == "add":
        # Check if the item already exists to prevent duplication
        if details not in target_list:
            target_list.append(details)
            st.toast(f"➕ Added {details.get('standard_name', target)} to {target.capitalize()}.", icon="✅")
        else:
             st.toast(f"❌ Item {details.get('standard_name', target)} already exists.", icon="❌")
             
    elif action == "remove":
        initial_length = len(target_list)
        # Filter out the item to be removed based on exact match of details
        record[target] = [item for item in target_list if item != details]
        
        if len(record[target]) < initial_length:
            st.toast(f"🗑️ Removed {details.get('standard_name', target)} from {target.capitalize()}.", icon="✅")
        else:
            st.toast(f"❌ Item not found in {target} for removal.", icon="❌")
            
    return record


# --- Gemini API Call Context (Relies on app.py for instruction) ---

def get_system_instruction(patient_name, record_context):
    return (
        "You are a Clinical Data Assistant. You can answer questions or modify the patient's record. You can also recommend additions, removals, or updates to their problems and medications based on best clinical practices. "
        f"The patient’s name is {patient_name}. When modifying data, return a JSON block with "
        "`action`, `target`, and `details` keys. Example: "
        '{"action": "add", "target": "medications", "details": {"standard_name": "Lisinopril", '
        '"standard_code_type": "RxNorm", "standard_code_value": "29046"}}\n'
        "--- PATIENT RECORD ---\n"
        f"{record_context}"
    )


# --- CHAT INTERFACE FUNCTION ---
def patient_chat_interface(patient_id, patient_name, record_context):
    """Chat UI with dynamic height based on interaction."""
    
    # NOTE: We dynamically import the external app helpers here
    from app import client, save_patient_data
    
    client = get_client_from_session()
    if client is None:
        st.warning("Cannot start chat — Gemini API key missing or invalid.")
        return

    # --- Setup ---
    chat_key = f"chat_history_{patient_id}"
    record_json = json.loads(record_context)
    
    # --- Initialize dynamic height state ---
    # The height calculation now moves to the rendering phase
    
    # Initialize chat history
    if chat_key not in st.session_state:
        st.session_state[chat_key] = [
            {
                "role": "assistant",
                "text": f"Hello! I’m your assistant for **{patient_name}** (ID: `{patient_id}`). "
                        "Ask about their conditions, medications, or suggest updates."
            }
        ]

    # --- DYNAMIC HEIGHT CALCULATION FIX ---
    
    # Calculate the number of messages (start at 1 to include the initial greeting)
    message_count = len(st.session_state[chat_key])
    
    # Determine the height: 
    # Start at 100px, add 70px per message, but cap at max_height (500px).
    MIN_HEIGHT = 20
    MAX_HEIGHT = 500
    HEIGHT_PER_MESSAGE = 80 
    
    # Calculate the desired height
    calculated_height = MIN_HEIGHT + (message_count * HEIGHT_PER_MESSAGE)
    
    # Apply the max height cap
    current_chat_height = min(calculated_height, MAX_HEIGHT)

    # 2. Anchor the entire chat area using st.empty()
    chat_placeholder = st.empty() 
    
    # 3. Use the DYNAMICALLY CALCULATED height
    chat_history_container = chat_placeholder.container(
        height=current_chat_height, # <-- Uses the calculated dynamic height
        border=True
    )

    with chat_history_container:
        for msg in st.session_state[chat_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["text"])
    
    # --- Input Bar and Logic ---
    user_prompt = st.chat_input(
        placeholder=f"Suggest an update or ask about {patient_name}'s record..."
    )
    
    if user_prompt:
        # --- PREPARATION ---
        prompt = user_prompt.strip()
        st.session_state[chat_key].append({"role": "user", "text": prompt})
        
        # The height calculation is automatic now—no need for explicit state update here.
        # FIX: Display the user message immediately for better UX
        with chat_history_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        # --- API CALL & RESPONSE PROCESSING (The Fix) ---
        with st.spinner("Assistant thinking..."):
            try:
                api_history = [
                    types.Content(
                        role=("user" if m["role"] == "user" else "model"),
                        parts=[types.Part(text=m["text"])]
                    )
                    for m in st.session_state[chat_key]
                ]

                config = types.GenerateContentConfig(
                    system_instruction=get_system_instruction(patient_name, record_context)
                )
                response = client.models.generate_content(
                    model=MODEL, contents=api_history, config=config
                )

                text = (response.text or "").strip() or "_No response received._"
                st.session_state[chat_key].append({"role": "assistant", "text": text})

                # --- 1. DETECT AND PARSE JSON MODIFICATION ---
                if "{" in text and "}" in text and ('add' in text or 'remove' in text):
                    try:
                        start_index = text.find("{")
                        end_index = text.rfind("}") + 1
                        candidate = text[start_index:end_index]
                        
                        mod = json.loads(candidate)
                        
                        act = mod.get("action", "").lower()
                        tgt = mod.get("target", "")
                        det = mod.get("details", {})

                        # Validate the core action/target
                        if act in ["add", "remove"] and tgt in ["problems", "medications"]:
                            
                            # --- 2. APPLY UPDATE TO SESSION STATE ---
                            latest_record = record_json
                            updated_record = update_record_data(latest_record, act, tgt, det)
                            
                            # --- 3. PERSIST CHANGES ---
                            st.session_state.patients[patient_id]['notes'][-1]['raw_data'] = updated_record
                            
                            if save_patient_data(patient_id, st.session_state.patients[patient_id]):
                                # Rerun to update the tables and height instantly
                                st.rerun() 
                            else:
                                st.warning("Update applied but failed to save to database.")
                        
                    except Exception:
                        st.toast("AI generated invalid JSON. Please rephrase the request.", icon="⚠️")

                # Rerun for standard text response (if no JSON update was applied)
                st.rerun() 

            except Exception as e:
                err = f"⚠️ Model error: {e}"
                st.error(err)
                st.session_state[chat_key].append({"role": "assistant", "text": err})
                st.rerun()