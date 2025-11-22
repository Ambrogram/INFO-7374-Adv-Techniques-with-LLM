"""
Constants for prompts, sheet names, and column names.
"""

# Full instruction string provided by the assignment
INSTR = (
    "Suppose that you are a medical diagnosis assistant, skilled in recognizing diseases and prescribing treatments. "
    "If a patient has the following condition, give me the most probable diagnosis and the related probability and urgency "
    "and rationale for the diagnosis besides suitable treatments. Start your response with Diagnosis: "
    "Now follow these additional critical rules strictly: "
    "On the FIRST line, output ONLY one final, definitive disease name after 'Diagnosis:' and do not add any other text on that line. "
    "Do NOT put explanation, reasoning, probability, urgency, or treatment information on the Diagnosis line. "
    "After the Diagnosis line, continue the response and provide the following sections in order: "
    "1) 'Probability: <percentage or qualitative level>', "
    "2) 'Urgency: <triage / how soon to act>', "
    "3) 'Rationale: <brief clinical reasoning>', "
    "4) 'Treatment: <suitable treatments or management>'. "
    "The diagnosis line must contain exactly one disease name in its full form, followed by its standard abbreviation in parentheses if applicable. "
    "The format of the first line must strictly be: 'Diagnosis: <Full Disease Name (Abbreviation)>' "
    "If uncertain, still output only the single most probable disease name in that format on the first line. "
    "Start your response with Diagnosis:\n"
)


# Target sheet names to process in the Test Data workbook
# Include common variants to be robust to naming
TARGET_SHEETS = [
    "NewSamples",
    "FineTunedSamples",
    "New Samples",
    "Finetuned Samples",
]

# Column names
COL_CASE_NO = "CaseNO"
COL_QUERY = "Patient Conditions"
COL_CORRECT_DIAG = "Correct Diagnosis"
COL_CORRECT_TREAT = "Correct Treatment"

# Before fine-tuning columns
COL_RESP_BEFORE = "Response of Engine Before Fine-tuning"
COL_RESULT_BEFORE = "Result of Diagnosis (Correct /Wrong ) Before Fine-tuning"

# After fine-tuning columns (kept here for completeness/extension)
COL_RESP_AFTER = "Response of Engine After Fine-tuning"
COL_RESULT_AFTER = "Result of Diagnosis (Correct /Wrong ) After Fine-tuning"

# Helper column for debugging parsed diagnosis
COL_PARSED_HELPER_BEFORE = "Diagnosis Parsed (Before)"
COL_PARSED_HELPER_AFTER = "Diagnosis Parsed (After)"


