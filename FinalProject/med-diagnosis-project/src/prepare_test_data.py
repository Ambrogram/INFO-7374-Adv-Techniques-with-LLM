"""
Prepare test data by prepending the medical diagnosis instruction to the
'Patient Conditions' column for the target sheets, and save to processed data.
"""
from __future__ import annotations

from config.settings import TEST_DATA_XLSX, PROCESSED_TEST_DATA_XLSX
from src.constants import TARGET_SHEETS, COL_QUERY
from src.prompt_utils import prepend_instruction
from src.excel_utils import read_workbook, write_workbook, transform_patient_conditions


def main() -> None:
    sheets = read_workbook(TEST_DATA_XLSX)
    transformed = transform_patient_conditions(
        sheets=sheets,
        target_sheet_names=TARGET_SHEETS,
        column_name=COL_QUERY,
        prepend_func=prepend_instruction,
    )
    write_workbook(PROCESSED_TEST_DATA_XLSX, transformed)
    print(f"[DONE] Saved processed workbook to: {PROCESSED_TEST_DATA_XLSX}")


if __name__ == "__main__":
    main()


