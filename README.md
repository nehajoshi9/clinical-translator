# Gemini Clinical Synthesizer

An AI-powered Streamlit app that automatically **extracts, standardizes, and maintains clinical patient records** using Google’s **Gemini 2.5 Flash** model.  
It synthesizes uploaded medical notes into structured JSON, displays them as interactive data tables, and enables a **conversational assistant** to query or update patient records intelligently — all backed by **Firebase Firestore**.

---

## Inspiration

Healthcare data is scattered across formats — handwritten notes, scanned PDFs, EMRs, and reports — making it difficult for clinicians to access unified, structured insights.
We wanted to explore how **LLMs can bridge the gap between unstructured notes and actionable data**, and how conversational AI could streamline reviewing and updating patient histories.
To do so, we learned about SNOMED and RxNorm codes and how they are used across the healthcare industry. Thus, the handwritten notes are being converted into these standardized codes. 

---

## What it does

- Upload any clinical document (PNG, JPG, or PDF)  
- Gemini performs **OCR + clinical entity extraction**, producing a structured JSON summary  
- Data is displayed in **interactive tables** (problems, medications, summary)  
- A **conversational assistant** lets users:
  - Ask questions about patient data  
  - Add or remove problems/medications  
  - Update the quick summary dynamically  
- All updates are stored in **Firestore**, ensuring persistence across sessions

---

## How we built it

- **Frontend:** Streamlit (Python)
- **AI Model:** Google Gemini 2.5 Flash via `google-genai` SDK  
- **Database:** Google Cloud Firestore for persistent patient storage  
- **Data Modeling:** Pydantic schemas for standardized structure  
- **OCR & Extraction:** Handled natively through Gemini multimodal input  
- **Realtime Chat:** Streamlit’s `st.chat_message` and `st.chat_input` with custom logic for JSON-based data modification

---

## Setup

1. Clone this repo  
   ```bash
   git clone https://github.com/yourusername/gemini-clinical-synthesizer.git
   cd gemini-clinical-synthesizer
2. Install dependencies

3. pip install -r requirements.txt


4. Add a .env file

GEMINI_API_KEY=your_google_genai_key
GCP_SERVICE_ACCOUNT_FILE=path/to/serviceAccount.json


5. Run the app
```bash
streamlit run app.py
