# app.py

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
# Import the necessary chat interface function
from patient_chatbot import patient_chat_interface, update_record_data 

# -------------------------------
# 1. SCHEMA
# -------------------------------

class ClinicalTerm(BaseModel):
    standard_name: str
    standard_code_type: str
    standard_code_value: str

class ClinicalTranslation(BaseModel):
    patient_id: str
    date_of_service: str
    quick_summary: str
    problems: List[ClinicalTerm]
    medications: List[ClinicalTerm]

TargetSchema = ClinicalTranslation

# -------------------------------
# 2. CONFIG
# -------------------------------

st.set_page_config(page_title="Clinical Data Synthesizer", layout="wide")
load_dotenv()

MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("FATAL ERROR: Missing GEMINI_API_KEY in .env file.")
    st.stop()

# -------------------------------
# 3. CACHED CLIENTS (INSTANT)
# -------------------------------

# NOTE: The Firestore/Gemini client initialization logic should remain the same
# as defined in the provided app.py snippet.
@st.cache_resource
def get_firestore_client():
    try:
        path = os.getenv("GCP_SERVICE_ACCOUNT_FILE")
        if path and os.path.exists(path):
            return firestore.Client.from_service_account_json(path)
        return firestore.Client()
    except Exception:
        return None

@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)

db = get_firestore_client()
client = get_gemini_client()

# -------------------------------
# 4. GEMINI SYNTHESIS
# -------------------------------

SYSTEM_INSTRUCTION = (
    "You are a Clinical Data Synthesizer. Perform OCR on all uploaded clinical documents, "
    "then merge and standardize all problems and medications into one JSON summary."
)

@st.cache_data(show_spinner=False)
def translate_clinical_note_multi(uploaded_files, schema):
    contents = []
    for file in uploaded_files:
        file.seek(0)
        img = Image.open(io.BytesIO(file.read()))
        contents.append(img)
    contents.append("Extract all problems and medications and synthesize a unified JSON summary.")

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

# -------------------------------
# 5. FIRESTORE HELPERS
# -------------------------------

@st.cache_data(show_spinner=False)
def load_patients_from_firestore():
    if not db:
        return {}
    try:
        return {doc.id: doc.to_dict() for doc in db.collection("patients").stream()}
    except Exception:
        return {}

def save_patient_data(pid, data):
    if not db:
        return False
    try:
        db.collection("patients").document(pid).set(data)
        return True
    except Exception:
        return False

# -------------------------------
# 6. STATE
# -------------------------------

def initialize_state():
    if "patients" not in st.session_state:
        st.session_state.patients = load_patients_from_firestore()
    if "page" not in st.session_state:
        st.session_state.page = "dashboard"
    if "current_patient_id" not in st.session_state:
        st.session_state.current_patient_id = None

    if not st.session_state.patients:
        demo = {
            "P-1001": {"name": "Jane Doe", "date_added": "2025-10-01", "notes": []},
            "P-1002": {"name": "John Smith", "date_added": "2025-10-15", "notes": []},
        }
        st.session_state.patients.update(demo)
        for pid, pdata in demo.items():
            save_patient_data(pid, pdata)

# -------------------------------
# 7. DASHBOARD (Fixed width)
# -------------------------------

def patient_dashboard():
    st.title("Patient Notes Dashboard")
    st.markdown("---")

    st.caption("Connected to Firestore." if db else "Firestore unavailable — local only.")

    c1, c2 = st.columns([1, 2])

    with c1:
        st.subheader("Add New Patient")
        with st.form("add_patient", clear_on_submit=True):
            name = st.text_input("Full Name")
            submit = st.form_submit_button("Add Patient")
            if submit and name:
                pid = f"P-{len(st.session_state.patients) + 1001}"
                pdata = {"name": name, "date_added": datetime.date.today().strftime("%Y-%m-%d"), "notes": []}
                st.session_state.patients[pid] = pdata
                save_patient_data(pid, pdata)
                st.success(f"Added {name} (ID: {pid})")

    with c2:
        st.subheader("Current Patients")
        pats = st.session_state.patients
        if not pats:
            st.info("No patients yet.")
            return
            
        for pid, p in sorted(pats.items(), key=lambda x: x[1]["name"]):
            card = st.container(border=True)
            cols = card.columns([3, 1])
            cols[0].markdown(f"**{p['name']}**<br><small>ID: `{pid}` | Notes: {len(p['notes'])}</small>", unsafe_allow_html=True)
            
            if cols[1].button("View →", key=f"view_{pid}", width='stretch'):
                st.session_state.page = "details"
                st.session_state.current_patient_id = pid
                st.rerun()

# -------------------------------
# 8. DETAIL PAGE (Anchored Layout Fix)
# -------------------------------

def clinical_translator(pid):
    pdata = st.session_state.patients.get(pid)
    if not pdata:
        st.error("Patient not found.")
        return

    # --- Top Navigation ---
    if st.button("← Back", type="secondary"):
        st.session_state.page = "dashboard"
        st.session_state.current_patient_id = None
        st.rerun()

    # --- Title / Info ---
    st.title(f"Patient: {pdata['name']}")
    st.caption(f"ID: `{pid}` • Added {pdata['date_added']}")
    st.markdown("---")

    # --- Columns (Anchored Layout Fix: Content rendered directly inside columns) ---
    left_col, right_col = st.columns([1.1, 1])
    
    # --- LEFT COLUMN: Uploader and Chat ---
    with left_col:
        st.subheader("Upload Clinical Documents")
        uploaded_files = st.file_uploader(
            "Upload notes for this patient",
            type=["png", "jpg", "jpeg", "pdf"],
            accept_multiple_files=True,
            key=f"upload_{pid}"
        )

        if uploaded_files and st.button("Synthesize & Save", type='primary', width='stretch'):
            with st.spinner("Synthesizing..."):
                try:
                    result = translate_clinical_note_multi(uploaded_files, TargetSchema)
                    parsed = json.loads(result)
                    new_note = {
                        "date_of_service": parsed.get("date_of_service", datetime.date.today().strftime("%Y-%m-%d")),
                        "summary": parsed.get("quick_summary", "N/A"),
                        "raw_data": parsed
                    }
                    pdata["notes"].append(new_note)
                    st.session_state.patients[pid] = pdata
                    save_patient_data(pid, pdata)
                    st.success("Record saved successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during synthesis: {e}")

        st.markdown("---")
        st.subheader("Conversational Assistant")

        # Chat renders immediately
        notes = pdata["notes"]
        if not notes:
            st.info("Synthesize a record first to enable the chat assistant.")
        else:
            latest = notes[-1]
            record_json = json.dumps(latest["raw_data"], indent=2)
            patient_chat_interface(pid, pdata["name"], record_json)


    # --- RIGHT COLUMN: Summary Tables (Read-Only Data) ---
    with right_col:
        st.subheader("Unified Record Summary")

        notes = pdata["notes"]
        if not notes:
            st.info("No synthesized data yet.")
            return

        latest = notes[-1]
        data = latest["raw_data"]
        st.info(f"**Synthesis Date:** {latest['date_of_service']}  \n**Summary:** {data.get('quick_summary')}")

        st.markdown("#### Conditions & Problems")
        COLUMN_NAMES = ['Standard Name', 'Code Type', 'Code Value']

        # Problems Table
        df_p = pd.DataFrame(data.get("problems", []))
        try: df_p.columns = COLUMN_NAMES
        except ValueError: df_p = pd.DataFrame(columns=COLUMN_NAMES)
        st.dataframe(df_p, width='stretch', hide_index=True)

        st.markdown("#### Medications")
        # Medications Table
        df_m = pd.DataFrame(data.get("medications", []))
        try: df_m.columns = COLUMN_NAMES
        except ValueError: df_m = pd.DataFrame(columns=COLUMN_NAMES)
        st.dataframe(df_m, width='stretch', hide_index=True)

        if len(notes) > 1:
            with st.expander("View Record History"):
                st.dataframe(pd.DataFrame(notes), width='stretch', hide_index=True)



# -------------------------------
# 9. MAIN
# -------------------------------

def main():
    initialize_state()
    if st.session_state.page == "dashboard":
        patient_dashboard()
    elif st.session_state.page == "details":
        clinical_translator(st.session_state.current_patient_id)

if __name__ == "__main__":
    main()