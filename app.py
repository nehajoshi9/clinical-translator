import streamlit as st
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
from PIL import Image
import io
import pandas as pd # Essential for DataFrame output

# --- 1. PYDANTIC SCHEMA DEFINITION ---

class ClinicalTerm(BaseModel):
    """Represents a single clinical entity extracted and standardized."""
    # Removed 'source_text' for cleaner, standardized output
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

load_dotenv() 
st.set_page_config(layout="wide", page_title="Clinical Data Synthesizer")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
MODEL = "gemini-2.5-flash" # Stable and fast model for multimodal tasks

if not GEMINI_API_KEY:
    st.error("FATAL ERROR: GEMINI_API_KEY not found. Please create a .env file and add your key.")
    st.stop()
    
client = genai.Client(api_key=GEMINI_API_KEY)


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


# --- 5. STREAMLIT USER INTERFACE (FINAL VERSION) ---

def main():
    st.title("🏥 Cross-System Clinical Data Synthesizer")
    st.markdown("Analyzes **multiple handwritten notes** and synthesizes a single, unified, standards-compliant clinical record.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload All Patient Documents")
        
        uploaded_files = st.file_uploader(
            "Upload all notes for ONE patient (e.g., Progress Note, Discharge Summary, Lab)", 
            type=["png", "jpg", "jpeg", "pdf"],
            accept_multiple_files=True
        )

        st.markdown(f"**Files Uploaded:** {len(uploaded_files)}")
        
        if uploaded_files and st.button("✨ Synthesize & Standardize Record", use_container_width=True):
            with st.spinner(f"Synthesizing data from {len(uploaded_files)} documents..."):
                try:
                    json_output_text = translate_clinical_note_multi(uploaded_files, TargetSchema)
                    data = json.loads(json_output_text)

                    with col2:
                        st.subheader("2. Unified Standardized Record")
                        st.success("✅ Synthesis Complete! Comprehensive record created from all notes.")
                        
                        # --- Display Summary ---
                        st.info(f"**Patient ID:** {data.get('patient_id')} | **Most Recent Date:** {data.get('date_of_service')}")
                        st.write(f"**Synthesized Summary:** {data.get('quick_summary')}")
                        
                        # --- Column Names for DataFrame ---
                        COLUMN_NAMES = ['Standard Name', 'Code Type', 'Code Value']

                        # --- Display Problems (Data Table) ---
                        st.markdown("#### Conditions & Problems (SNOMED CT) 📋")
                        
                        # Use a try/except or conditional logic in a clean way
                        try:
                            df_problems = pd.DataFrame(data['problems'])
                            df_problems.columns = COLUMN_NAMES
                        except ValueError:
                            # Handles Length Mismatch error if list is empty
                            df_problems = pd.DataFrame(columns=COLUMN_NAMES)
                        
                        st.dataframe(df_problems, use_container_width=True)

                        # --- Display Medications (Data Table) ---
                        st.markdown("#### Medications (RxNorm) 💊")
                        
                        try:
                            df_meds = pd.DataFrame(data['medications'])
                            df_meds.columns = COLUMN_NAMES
                        except ValueError:
                            # Handles Length Mismatch error if list is empty
                            df_meds = pd.DataFrame(columns=COLUMN_NAMES)
                            
                        st.dataframe(df_meds, use_container_width=True)

                except Exception as e:
                    with col2:
                        # Catch API and general exceptions
                        st.error(f"An error occurred during synthesis: {e}")
                        st.warning("Ensure all uploaded files are images/PDFs and contain legible text.")


if __name__ == "__main__":
    main()