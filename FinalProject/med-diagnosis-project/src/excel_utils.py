"""
Excel I/O utilities used in the pipeline.
"""
from __future__ import annotations
from typing import Dict, Callable
import pandas as pd


def read_workbook(path: str | bytes | "os.PathLike[str]") -> Dict[str, pd.DataFrame]:
    """
    Read an Excel workbook into a dict of DataFrames by sheet name.
    """
    xls = pd.ExcelFile(path)
    data: Dict[str, pd.DataFrame] = {}
    for name in xls.sheet_names:
        data[name] = pd.read_excel(path, sheet_name=name)
    return data


def write_workbook(path: str | bytes | "os.PathLike[str]", sheets: Dict[str, pd.DataFrame]) -> None:
    """
    Write a mapping of sheet name -> DataFrame to an Excel workbook.
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


def transform_patient_conditions(
    sheets: Dict[str, pd.DataFrame],
    target_sheet_names: list[str],
    column_name: str,
    prepend_func: Callable[[str], str],
) -> Dict[str, pd.DataFrame]:
    """
    For each target sheet, prepend the instruction to the 'Patient Conditions' column.
    Non-target sheets are returned unchanged.
    """
    out: Dict[str, pd.DataFrame] = {}
    for name, df in sheets.items():
        if name in target_sheet_names:
            if column_name not in df.columns:
                raise KeyError(f"Sheet '{name}' does not have a '{column_name}' column.")
            series = df[column_name].astype("object").fillna("")
            df[column_name] = [prepend_func(val) for val in series]
            out[name] = df
        else:
            out[name] = df
    return out


