from __future__ import annotations
from pathlib import Path
import pandas as pd
from .config import settings


def load_sources(data_dir: str | Path,
                 housing_xlsx: str = "rmit_housing_full.xlsx",
                 emergency_csv: str = "emergency.csv",
                 oshc_csv: str = "oshc.csv") -> pd.DataFrame:
    data_path = Path(data_dir)

    housing_path = data_path / housing_xlsx
    emergency_path = data_path / emergency_csv
    oshc_path = data_path / oshc_csv

    housing_df = pd.read_excel(housing_path, sheet_name=settings.sheet_name)
    housing_df["source"] = "housing"

    emergency_df = pd.read_csv(emergency_path)
    emergency_df["source"] = "emergency"

    oshc_df = pd.read_csv(oshc_path)
    oshc_df["source"] = "oshc"

    combined = pd.concat([emergency_df, oshc_df, housing_df], ignore_index=True)
    return combined


def dataframe_to_documents(df: pd.DataFrame) -> list[str]:
    return df.astype(str).apply(lambda row: " | ".join(row.values), axis=1).tolist()
