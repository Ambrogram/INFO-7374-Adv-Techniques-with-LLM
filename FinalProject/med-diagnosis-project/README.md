## Med Diagnosis Fine-tuning Pipeline

This project organizes a small but complete workflow to evaluate and fine-tune a medical diagnosis LLM following the course assignment steps. It is structured like a production ML pipeline with clear modules and paths.

### Features
- Clean modular code under `src/` with reusable helpers
- Centralized configuration and paths under `config/`
- Environment variables via `.env`
- Clear data locations: `data/raw`, `data/processed`, `data/results`
- Baseline inference pipeline before fine-tuning
- Easily extensible for fine-tuning (placeholders included)

### Repository Layout
```
med-diagnosis-project/
├── README.md
├── .env
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── settings.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
├── src/
│   ├── __init__.py
│   ├── constants.py
│   ├── prompt_utils.py
│   ├── excel_utils.py
│   ├── prepare_test_data.py
│   ├── run_baseline.py
│   └── fine_tune_utils.py
└── tests/
    └── test_prompt_utils.py
```

### 1) Environment Setup
1. Ensure Python 3.10+ is installed.
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt
```

3. Create `.env` at the project root with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
BASE_MODEL=gpt-4o
```

### 2) Data Placement
- Put the provided `Test_Data.xlsx` into `data/raw/`.
- Baseline results will be written to `data/results/Test_Data_before_ft_filled.xlsx`.
- The processed file with instructions will be written to `data/processed/Test_Data_with_instr.xlsx`.

### 3) Prepare Test Data (prepend instructions)
This step prepends the medical diagnostic instruction to the “Patient Conditions” column in the target sheets.

```bash
python -m src.prepare_test_data
```

Outputs: `data/processed/Test_Data_with_instr.xlsx`

### 4) Run Baseline (before fine-tuning)
This step runs the base model over the processed workbook and fills in the “Before Fine-tuning” response and correctness columns.

```bash
python -m src.run_baseline
```

Outputs: `data/results/Test_Data_before_ft_filled.xlsx`

### 5) Fine-tuning (placeholders)
- Convert Fine-tuning/Validation datasets to JSONL (to be implemented in `src/fine_tune_utils.py`).
- Upload and fine-tune at the OpenAI platform; record the loss chart.

### 6) After Fine-tuning
- Update the model name in `.env` (e.g., your fine-tuned model ID) and reuse a similar script to fill the “After Fine-tuning” columns (you can extend `src/run_baseline.py` or add a sibling script).

### Testing
Simple unit test example:
```bash
pytest -q
```

### Notes
- Do not hardcode paths. Use `config/settings.py` for path management.
- Constants such as sheet names and column names live in `src/constants.py`.
- The project is designed to be extended for JSONL generation and hyperparameter logging.


