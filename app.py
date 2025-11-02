import streamlit as st
from google import genai
from google.genai import types
from schema import ClinicalTranslatorOutput # Import your Pydantic schema
import os
import json

# --- 1. Configuration ---
st.set_page_config(layout="wide", page_title="Clinical Translator")

# Ensure the API key is set
if "GEMINI_API_KEY" not in os.environ:
    st.error("Please set the GEMINI_API_KEY environment variable.")
    st.stop()

# Initialize the Gemini Client
client = genai.Client()
MODEL = "gemini-2.5-pro" # Use a powerful model for complex reasoning and coding

# --- 2. Prompt Definition (The CRITICAL STEP) ---

# Define the System Instruction for the AI's role
SYSTEM_INSTRUCTION = (
    "You are an expert clinical terminology mapper. Your task is to extract all Problem and Medication terms "
    "from the following clinical document and map them to their corresponding **SNOMED CT** and **RxNorm** codes. "
    "You MUST strictly adhere to the provided JSON schema."
)

# Define a Few-Shot Example (Crucial for high accuracy and style)
FEW_SHOT_EXAMPLE = """
**Input Note:**
Pt is a 65 y/o male with longstanding HTN and DM who has been taking Metformin qd. He came in today complaining of SOB.

**Expected Output JSON (as per schema):**
{
  "patient_summary": {
    "patient_id": "P-4321",
    "source_doc_type": "Progress Note",
    "date_of_service": "2025-11-02"
  },
  "standardized_problem_list": [
    {
      "source_term": "HTN",
      "standard_name": "Essential (primary) hypertension",
      "problem_code": "38341003"
    },
    {
      "source_term": "DM",
      "standard_name": "Diabetes mellitus",
      "problem_code": "73211009"
    },
    {
      "source_term": "SOB",
      "standard_name": "Shortness of breath",
      "problem_code": "267036007"
    }
  ],
  "standardized_medication_list": [
    {
      "source_term": "Metformin qd",
      "drug_name": "Metformin",
      "rxnorm_code": "6809"
    }
  ]
}
"""

# --- 3. Streamlit Interface and Logic ---

st.title("🏥 Cross-System Clinical Translator")
st.markdown("A **Semantic Bridge** for converting messy clinical narrative into machine-readable SNOMED CT and RxNorm codes.")

with st.expander("📝 Few-Shot Example & Instructions", expanded=False):
    st.code(FEW_SHOT_EXAMPLE, language='markdown')
    st.caption("This example guides Gemini on the style and required codes.")

input_note = st.text_area(
    "Paste Unstructured Clinical Note Here:",
    height=200,
    value="Patient (Pt) presents with persistent H/A for 3 days. History of taking Lipitor for high cholesterol. Vitals show high BP. Plan: check full labs."
)

if st.button("✨ Translate to Structured JSON", use_container_width=True):
    if not input_note.strip():
        st.error("Please enter a clinical note to translate.")
    else:
        # Create the full prompt message including the example and the new input
        full_prompt = (
            f"Here is a few-shot example for context:\n{FEW_SHOT_EXAMPLE}\n\n"
            f"Now, process the following new clinical note and provide ONLY the standardized JSON output:\n\n"
            f"**New Clinical Note:**\n{input_note}"
        )
        
        with st.spinner("Mapping clinical terms and generating codes..."):
            try:
                # Call the Gemini API with structured output enforcement
                response = client.models.generate_content(
                    model=MODEL,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        response_schema=ClinicalTranslatorOutput,
                    )
                )

                # The response text is the valid JSON string
                json_output = response.text
                
                # Display the result
                st.success("✅ Translation Complete!")
                st.subheader("Standardized, Code-Compliant JSON Output")
                
                # Display as formatted JSON for easy viewing
                st.json(json.loads(json_output))
                
                # Optionally, display the raw JSON for copying
                with st.expander("View Raw JSON String"):
                    st.code(json_output, language='json')

            except Exception as e:
                st.error(f"An error occurred during translation: {e}")
                st.caption("Check your API key and ensure the input is clear.")