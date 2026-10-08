from __future__ import annotations
from pathlib import Path
import sqlite3
import zipfile
import pandas as pd

DISPLAY_COLS = [
    "LoanNumber","BorrowerName","BorrowerCity","BorrowerState","NAICSCode","BusinessType",
    "OriginatingLender","CurrentApprovalAmount","JobsReported","ForgivenessAmount",
    "NotForgivenAmount","ForgivenPercentage","PROCEED_Per_Job","DateApproved"
]

def load_sqlite(db_path: str | Path) -> pd.DataFrame:
    with sqlite3.connect(str(db_path)) as conn:
        return pd.read_sql_query("SELECT * FROM loans_scored", conn)

def load_csv_or_zip(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith('.csv')]
            if not csvs:
                raise ValueError("ZIP has no CSV file")
            with z.open(csvs[0]) as f:
                return pd.read_csv(f, usecols=lambda c: c in DISPLAY_COLS, nrows=nrows, low_memory=False)
    return pd.read_csv(path, usecols=lambda c: c in DISPLAY_COLS, nrows=nrows, low_memory=False)
