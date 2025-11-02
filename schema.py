from pydantic import BaseModel, Field, conlist
from typing import List

# Define the structure for a single problem
class StandardizedProblem(BaseModel):
    """Schema for a single mapped clinical problem."""
    source_term: str = Field(description="The original, messy term from the clinical note (e.g., 'High blood pressure').")
    standard_name: str = Field(description="The formal, standardized name (e.g., 'Hypertension').")
    problem_code: str = Field(description="The corresponding SNOMED CT ID (e.g., '38341003').")

# Define the structure for a single medication
class StandardizedMedication(BaseModel):
    """Schema for a single mapped medication."""
    source_term: str = Field(description="The original, messy medication reference (e.g., 'Take Lisinopril qd').")
    drug_name: str = Field(description="The generic drug name (e.g., 'Lisinopril').")
    rxnorm_code: str = Field(description="The unique RxNorm identifier for the drug.")

# Define the final, top-level output schema
class ClinicalTranslatorOutput(BaseModel):
    """The final structured output for the clinical translator."""
    patient_summary: dict = Field(description="Placeholder dictionary for patient metadata (ID, doc type, date).")
    standardized_problem_list: List[StandardizedProblem] = Field(description="A list of all identified and coded clinical problems.")
    standardized_medication_list: List[StandardizedMedication] = Field(description="A list of all identified and coded medications.")