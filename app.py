import streamlit as st
import os
import json
import pandas as pd 
import io
import datetime
import time 
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image 
from pydantic import BaseModel, Field
from typing import List 
from google.cloud import firestore 
from patient_chatbot import patient_chat_interface # <-- CHANGE 1: Import the new, integrated chat function

# --- 1. PYDANTIC SCHEMA DEFINITION (The Blueprint) ---

class ClinicalTerm(BaseModel):
    """Represents a single clinical entity extracted and standardized."""
    # Note: 'source_text' removed from schema
    standard_name: str = Field(..., description="The standardized, human-readable name of the entity (e.g., 'Hypertension', 'Lisinopril').")
    standard_code_type: str = Field(..., description="The type of code used (e.g., 'SNOMED_CT' for problems, 'RxNorm' for drugs).")
    standard_code_value: str = Field(..., description="The official, standardized code value.")


class ClinicalTranslation(BaseModel):
    """The final, structured output for the patient record, synthesized from all documents."""
    patient_id: str = Field(..., description="A unique identifier for the patient (e.g., P-4321).")
    date_of_service: str = Field(..., description="The date of the most recent note, in YYYY-MM-DD format.")
    quick_summary: str = Field(..., description="A single sentence summarizing the patient's main clinical problems and current status, synthesized from ALL documents.")
    
    problems: List[ClinicalTerm] = Field(..., description="A comprehensive list of ALL medical conditions/symptoms standardized to SNOMED CT.")
    medications: List[ClinicalTerm] = Field(..., description="A comprehensive list of ALL medications standardized to RxNorm.")

TargetSchema = ClinicalTranslation


# --- 2. CONFIGURATION & SETUP ---

st.set_page_config(
    page_title="Clinical Data Synthesizer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv() 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
MODEL = "gemini-2.5-flash"

if not GEMINI_API_KEY:
    st.error("FATAL ERROR: GEMINI_API_KEY not found. Please create a .env file and add your key.")
    st.stop()
    
client = genai.Client(api_key=GEMINI_API_KEY)


# --- FIRESTORE INITIALIZATION ---
# Robustly initialize Firestore client using Service Account or Default Credentials
try:
    # Look for an environment variable pointing to the Service Account JSON file path
    service_account_path = os.getenv("GCP_SERVICE_ACCOUNT_FILE")
    
    if service_account_path and os.path.exists(service_account_path):
        # Authenticate using a specific service account file (Best Practice for deployment)
        db = firestore.Client.from_service_account_json(service_account_path)
        FIRESTORE_STATUS = "Ready (Service Account)"
    else:
        # Authenticate using Application Default Credentials (ADC) or environment credentials
        db = firestore.Client()
        FIRESTORE_STATUS = "Ready (Default Credentials)"

except Exception as e:
    # Fallback status if initialization fails (e.g., missing credentials)
    db = None
    FIRESTORE_STATUS = f"Error: Firestore needs credentials. Details: {e}"


# --- 3. PROMPT DEFINITION ---

FEW_SHOT_GUIDE = """
GUIDE: You must perform OCR on ALL uploaded documents. Synthesize a single, comprehensive list of problems and medications across all notes, ensuring no duplicate entries. Use the most recent date of service found across all notes.
"""

SYSTEM_INSTRUCTION = (
    "You are an expert Clinical Data Synthesizer. Your task is to perform OCR/handwriting recognition on the multiple provided documents, "
    "and create a single, unified, standards-compliant summary of the patient's entire record. "
    "Note that the target JSON schema no longer requires the 'source_term' field. " 
    f"You MUST strictly adhere to the provided JSON schema. {FEW_SHOT_GUIDE}" 
)


# --- 4. CORE GEMINI TRANSLATION LOGIC ---

@st.cache_data(show_spinner=False)
def translate_clinical_note_multi(uploaded_files, schema):
    contents = []
    
    # 1. Add all uploaded images to the contents list
    for uploaded_file in uploaded_files:
        image_bytes = uploaded_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        contents.append(img)
    
    # 2. Add the final instruction prompt
    action_prompt = "Synthesize and extract ALL problems and ALL medications from these documents into the single JSON structure."
    contents.append(action_prompt)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=schema,
    )
    
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config
    )
    return response.text


# --- 5. FIRESTORE PERSISTENCE FUNCTIONS ---

@st.cache_data(show_spinner=False)
def load_patients_from_firestore(max_retries=5):
    """Loads all patient data from the Firestore 'patients' collection."""
    # Check status to prevent errors if db is None
    if db is None:
        return {}
    
    for attempt in range(max_retries):
        try:
            patients = {}
            # Retrieve all documents from the 'patients' collection
            patient_ref = db.collection("patients").stream()
            for doc in patient_ref:
                patients[doc.id] = doc.to_dict()
            return patients
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time) # Exponential backoff
            else:
                st.error(f"Failed to load data from Firestore after {max_retries} attempts: {e}")
                return {} # Return empty dictionary on failure
    return {}

def save_patient_data(patient_id, patient_data, max_retries=5):
    """Saves or updates a single patient's data in Firestore."""
    # Check status to prevent errors if db is None
    if db is None:
        st.error("Cannot save data: Firestore client is not initialized.")
        return False

    for attempt in range(max_retries):
        try:
            # Set the data for the specific patient document
            db.collection("patients").document(patient_id).set(patient_data)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time) # Exponential backoff
            else:
                st.error(f"Failed to save data to Firestore after {max_retries} attempts: {e}")
                return False # Return failure status
    return False


# --- 6. STREAMLIT APPLICATION PAGES ---

def initialize_state():
    """Initializes session state variables for navigation and data storage."""
    # Load persistent data from Firestore on first run
    if 'patients' not in st.session_state:
        # Use st.cache_data function to load data once
        st.session_state.patients = load_patients_from_firestore()
        
    if 'page' not in st.session_state:
        st.session_state.page = 'dashboard'
    if 'current_patient_id' not in st.session_state:
        st.session_state.current_patient_id = None
        
    # If no data loaded and connection is ready, initialize with a couple of demos
    if not st.session_state.patients and FIRESTORE_STATUS.startswith("Ready"):
        demo_patient_1 = {'name': 'Jane Doe', 'date_added': '2025-10-01', 'notes': []}
        demo_patient_2 = {'name': 'John Smith', 'date_added': '2025-10-15', 'notes': []}
        
        # Save demo patients to both session state and database
        st.session_state.patients['P-1001'] = demo_patient_1
        st.session_state.patients['P-1002'] = demo_patient_2
        save_patient_data('P-1001', demo_patient_1)
        save_patient_data('P-1002', demo_patient_2)


def patient_dashboard():
    """The 'Home Screen' showing the list of patients and the add form."""
    st.title("🏥 Patient Notes Dashboard")
    st.markdown("---")
    
    # NEW: Display Firestore connection status
    if FIRESTORE_STATUS.startswith("Ready"):
         st.success(f"✅ **Real Backend:** Connected to Firestore using: {FIRESTORE_STATUS.split('(')[1].replace(')', '')}")
    else:
         st.warning(f"⚠️ **Backend Status:** Firestore connection error. Data will not persist. Details: {FIRESTORE_STATUS}")
    
    col1, col2 = st.columns([1, 2])

    # --- Add New Patient Form ---
    with col1:
        st.subheader("Add New Patient")
        with st.form("add_patient_form", clear_on_submit=True):
            new_name = st.text_input("Patient Full Name")
            submitted = st.form_submit_button("➕ Add Patient")

            if submitted and new_name:
                new_id = f"P-{len(st.session_state.patients) + 1001}"
                new_patient_data = {
                    'name': new_name,
                    'date_added': datetime.date.today().strftime("%Y-%m-%d"),
                    'notes': [] 
                }
                
                # Save to Firestore
                if save_patient_data(new_id, new_patient_data):
                    st.session_state.patients[new_id] = new_patient_data
                    st.success(f"Patient {new_name} added and saved to database with ID: {new_id}")
                else:
                    st.error("Could not save patient to database. Try again.")


    # --- Patient List ---
    with col2:
        st.subheader(f"Current Patients ({len(st.session_state.patients)})")
        
        if not st.session_state.patients:
            st.info("No patients found. Add one on the left to get started!")
        else:
            sorted_patients = dict(sorted(st.session_state.patients.items(), key=lambda item: item[1]['name']))
            
            for p_id, p_data in sorted_patients.items():
                card = st.container(border=True)
                card_cols = card.columns([3, 1])

                card_cols[0].markdown(f"**{p_data['name']}** <br> <small>ID: `{p_id}` | Notes: **{len(p_data['notes'])}**</small>", unsafe_allow_html=True)

                if card_cols[1].button("View Notes →", key=f"view_{p_id}", use_container_width=True):
                    st.session_state.current_patient_id = p_id
                    st.session_state.page = 'details'
                    st.rerun()


def clinical_translator(patient_id):
    """The 'Details Screen' for processing notes for a selected patient."""
    
    patient_data = st.session_state.patients.get(patient_id)
    
    # Back button
    if st.button("← Back to Patient Dashboard", type="secondary"):
        st.session_state.page = 'dashboard'
        st.session_state.current_patient_id = None
        st.rerun()
        
    if not patient_data:
        st.error(f"Patient with ID {patient_id} not found.")
        st.session_state.page = 'dashboard'
        return
        
    st.title(f"📄 Note Processor for: {patient_data['name']}")
    st.markdown(f"**Patient ID:** `{patient_id}` | Date Added: {patient_data['date_added']}")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    # --- Document Upload & Processing (Multi-File Logic) ---
    with col1:
        st.subheader("1. Upload Clinical Documents")
        
        # --- Multi-File Uploader ---
        uploaded_files = st.file_uploader(
            "Upload ALL notes for this patient to generate a UNIFIED record.", 
            type=["png", "jpg", "jpeg", "pdf"],
            accept_multiple_files=True,
            key=f"uploader_{patient_id}"
        )
        
        st.info(f"**Files Selected:** {len(uploaded_files)}")
        
        # NOTE: Using type="primary" to make the main action button stand out (now blue)
        if uploaded_files and st.button("✨ Synthesize & Standardize Record", use_container_width=True):
            with st.spinner(f"Synthesizing data from {len(uploaded_files)} documents..."):
                try:
                    # Run the Core Function
                    json_output_text = translate_clinical_note_multi(uploaded_files, TargetSchema)
                    data = json.loads(json_output_text)

                    # 3. Store the processed note data (including the raw JSON)
                    new_note = {
                        'date_of_service': data.get('date_of_service', datetime.date.today().strftime("%Y-%m-%d")),
                        'summary': data.get('quick_summary', 'N/A'),
                        'raw_data': data 
                    }
                    # Always append the newest full synthesis to the list of notes
                    st.session_state.patients[patient_id]['notes'].append(new_note)

                    # Save updated patient data to Firestore
                    if save_patient_data(patient_id, st.session_state.patients[patient_id]):
                        st.success(f"✅ Record Synthesized and Saved to database for {patient_data['name']}!")
                    else:
                        st.warning("Record synthesized but failed to save to database.")
                    
                    # Rerun to update the history below
                    st.rerun() 

                except Exception as e:
                    st.error(f"An error occurred during synthesis: {e}")
                    st.warning("Please check the input files and ensure your API key is functional.")

    # --- Processed Note History & Display ---
    with col2:
        st.subheader("2. Unified Record & Detail")
        
        notes = st.session_state.patients[patient_id]['notes']
        
        if not notes:
            st.info("No synthesized records found. Upload notes on the left to create the first record.")
            
            # --- CHAT INTEGRATION POINT: No data yet, so show minimal chat ---
            st.markdown("---")
            with st.expander("💬 Conversational Assistant (Requires Synthesized Record)", expanded=False):
                # Pass minimal context if no notes exist
                patient_chat_interface(patient_id, patient_data['name'], "No synthesized data available yet. Please upload notes first.")
            return

        # Show the most recently added note
        latest_note = notes[-1]
        data = latest_note['raw_data']
        # COLUMN_NAMES updated to reflect the removal of 'source_text' from the schema
        COLUMN_NAMES = ['Standard Name', 'Code Type', 'Code Value']

        # --- Display Summary ---
        st.info(f"**Synthesis Date:** {latest_note['date_of_service']} \n **Record Summary:** {data.get('quick_summary')}")
        
        # --- Display Problems (Data Table) ---
        st.markdown("#### Conditions & Problems (SNOMED CT) 📋")
        
        try:
            df_problems = pd.DataFrame(data['problems'])
            df_problems.columns = COLUMN_NAMES
        except ValueError:
            # Handles Length Mismatch error if list is empty
            df_problems = pd.DataFrame(columns=COLUMN_NAMES)
        
        st.dataframe(df_problems, use_container_width=True, hide_index=True)

        # --- Display Medications (Data Table) ---
        st.markdown("#### Medications (RxNorm) 💊")
        
        try:
            df_meds = pd.DataFrame(data['medications'])
            df_meds.columns = COLUMN_NAMES
        except ValueError:
            # Handles Length Mismatch error if list is empty
            df_meds = pd.DataFrame(columns=COLUMN_NAMES)
            
        st.dataframe(df_meds, use_container_width=True, hide_index=True)
        
        # Display Raw JSON for export/debugging
        with st.expander("View Raw JSON Output"):
            st.code(json.dumps(data, indent=2), language='json')
            
        # Optional: Add an expander to see the list of all processed records
        if len(notes) > 1:
             with st.expander("View Full Synthesis History"):
                 st.dataframe(pd.DataFrame(notes), use_container_width=True)

    # --- CHAT INTEGRATION POINT: Full context is available here ---
    st.markdown("---")
    with st.expander("💬 Conversational Assistant for This Patient", expanded=False):
        # Pass the latest record JSON as context for the chat model
        latest_record_json = json.dumps(data, indent=2)
        patient_chat_interface(patient_id, patient_data['name'], latest_record_json)


def main():
    """Main control flow for the Streamlit application."""
    initialize_state()
    
    # The navigation now routes only between the Dashboard (list) and Details (translator + chat)
    
    if st.session_state.page == 'dashboard':
        patient_dashboard()
    elif st.session_state.page == 'details':
        clinical_translator(st.session_state.current_patient_id)


if __name__ == "__main__":
    main()
