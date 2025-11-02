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
import pandas as pd # <--- NEW: Import Pandas

# --- 1. PYDANTIC SCHEMA DEFINITION (The Blueprint) ---

# Define the individual term structure
class ClinicalTerm(BaseModel):
    """Represents a single clinical entity extracted and standardized."""
    source_text: str = Field(..., description="The exact term or phrase extracted from the source document (e.g., 'Pt has HTN').")
    standard_name: str = Field(..., description="The standardized, human-readable name of the entity (e.g., 'Hypertension', 'Lisinopril').")
    standard_code_type: str = Field(..., description="The type of code used (e.g., 'SNOMED_CT' for problems, 'RxNorm' for drugs).")
    standard_code_value: str = Field(..., description="The official, standardized code value.")


# Define the Final Document Summary (The output schema)
class ClinicalTranslation(BaseModel):
    """The final, structured output for the patient record, ready for EHR integration."""
    patient_id: str = Field(..., description="A unique identifier for the patient (e.g., P-4321).")
    date_of_service: str = Field(..., description="The date the note was written, in YYYY-MM-DD format.")
    quick_summary: str = Field(..., description="A single sentence summarizing the patient's main clinical problems and current status.")
    problems: List[ClinicalTerm] = Field(..., description="A list of all medical conditions and symptoms found in the note, standardized to SNOMED CT.")
    medications: List[ClinicalTerm] = Field(..., description="A list of all medications found in the note, standardized to RxNorm.")

TargetSchema = ClinicalTranslation


# --- 2. CONFIGURATION & SETUP ---

load_dotenv() 
st.set_page_config(layout="wide", page_title="Cross-System Clinical Translator")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
MODEL = "gemini-2.5-flash"

if not GEMINI_API_KEY:
    st.error("FATAL ERROR: GEMINI_API_KEY not found. Please create a .env file and add your key.")
    st.stop()
    
client = genai.Client(api_key=GEMINI_API_KEY)


# --- 3. PROMPT DEFINITION (Same as before) ---

FEW_SHOT_GUIDE = """
EXAMPLE GUIDE:
When processing the input, generate your output JSON based on this example. Note the standardization of abbreviations (HTN, DM, SOB) and the required codes (SNOMED_CT, RxNorm). 
Input Example: Pt is a 65 y/o male with longstanding HTN and DM who has been taking Metformin qd. He came in today complaining of SOB.
Expected Output (Schema structure only): 
{
  "patient_id": "P-4321", "date_of_service": "2025-11-02", 
  "quick_summary": "65 year old male with history of diabetes and hypertension presenting with shortness of breath.",
  "problems": [
    {"source_term": "HTN", "standard_name": "Essential (primary) hypertension", "standard_code_type": "SNOMED_CT", "standard_code_value": "38341003"},
    {"source_term": "DM", "standard_name": "Diabetes mellitus", "standard_code_type": "SNOMED_CT", "standard_code_value": "73211009"},
    {"source_term": "SOB", "standard_name": "Shortness of breath", "standard_code_type": "SNOMED_CT", "standard_code_value": "267036007"}
  ],
  "medications": [
    {"source_term": "Metformin qd", "standard_name": "Metformin", "standard_code_type": "RxNorm", "standard_code_value": "6809"}
  ]
}
"""

SYSTEM_INSTRUCTION = (
    "You are an expert Clinical Data Translator. Your task is to perform OCR/handwriting recognition on the provided image, "
    "extract the key problems and medications, and map them to standard codes (SNOMED_CT, RxNorm). "
    "You MUST strictly adhere to the provided JSON schema. "
    "Use clinical reasoning to infer the full meaning of abbreviations and correct any OCR errors. "
    f"--- FEW-SHOT INSTRUCTION GUIDE ---\n{FEW_SHOT_GUIDE}" 
)


# --- 4. CORE GEMINI TRANSLATION LOGIC (Same as before) ---

@st.cache_data(show_spinner=False)
def translate_clinical_note(uploaded_file, schema):
    image_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(image_bytes))

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=schema,
    )
    
    action_prompt = "Perform OCR on this entire document, then extract and standardize all problems and medications using the standardized codes."
    
    response = client.models.generate_content(
        model=MODEL,
        contents = [img, action_prompt],
        config=config
    )
    return response.text


# --- 5. STREAMLIT USER INTERFACE (UPDATED FOR DATAFRAMES) ---

def main():
    st.title("🏥 Cross-System Clinical Translator")
    st.markdown("Instantly convert **messy handwritten notes** into **standards-compliant JSON** via multimodal AI.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload Clinical Document")
        uploaded_file = st.file_uploader(
            "Upload Handwritten Patient Note (PNG, JPEG, or PDF)", 
            type=["png", "jpg", "jpeg", "pdf"]
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption='Document Ready for Translation', use_container_width=True)

        if st.button("✨ Translate & Standardize", type="primary", use_container_width=True, disabled=not uploaded_file):
            with st.spinner("1. Performing Handwriting OCR. 2. Mapping to SNOMED/RxNorm. 3. Generating DataFrames..."):
                try:
                    # 1. Run the Core Function (Get JSON text)
                    json_output_text = translate_clinical_note(uploaded_file, TargetSchema)
                    
                    # 2. Parse the JSON string into a Python dictionary
                    data = json.loads(json_output_text)

                    # 3. Process and Display Results
                    with col2:
                        st.subheader("2. Standardized Clinical Data")
                        st.success("✅ Translation Complete! Data is clean and ready for integration.")
                        
                        # --- Display Summary ---
                        st.info(f"**Patient ID:** {data.get('patient_id')} | **Date:** {data.get('date_of_service')}")
                        st.write(f"**Summary:** {data.get('quick_summary')}")
                        
                        # --- Display Problems (Data Table) ---
                        st.markdown("#### Conditions & Problems (SNOMED CT) 📋")
                        if data.get('problems'):
                            # Flatten the list of Pydantic objects into a DataFrame
                            df_problems = pd.DataFrame(data['problems'])
                            # Rename columns for clarity in the UI
                            df_problems.columns = ['Source Term', 'Standard Name', 'Code Type', 'Code Value']
                            st.dataframe(df_problems, use_container_width=True)
                        else:
                            st.markdown("*No problems identified.*")

                        # --- Display Medications (Data Table) ---
                        st.markdown("#### Medications (RxNorm) 💊")
                        if data.get('medications'):
                            df_meds = pd.DataFrame(data['medications'])
                            df_meds.columns = ['Source Term', 'Standard Name', 'Code Type', 'Code Value']
                            st.dataframe(df_meds, use_container_width=True)
                        else:
                            st.markdown("*No medications identified.*")

                        # Display Raw JSON for export/debugging
                        with st.expander("View Raw JSON String"):
                            st.code(json_output_text, language='json')

                except Exception as e:
                    with col2:
                        st.error(f"An error occurred during translation: {e}")
                        st.warning("Please check the input image and verify your API key.")

'''
    # Show the few-shot guide in an expander for reference/debugging
    with st.expander("🛠️ Internal Few-Shot Guide (For Reference Only)", expanded=False):
        st.code(FEW_SHOT_GUIDE, language='markdown')
'''
if __name__ == "__main__":
    main()