from __future__ import annotations
from pathlib import Path
import pandas as pd


# --- Q&A ingestion helpers ---
def load_qa_csv(csv_path: str | Path,
                question_col: str = "question",
                answer_col: str = "answer",
                source_label: str = "qa") -> pd.DataFrame:
    """Load a Q&A CSV into a dataframe.

    Parameters:
    - csv_path: path to the CSV file containing Q&A rows
    - question_col: name of the question column (default: "question")
    - answer_col: name of the answer column (default: "answer")
    - source_label: sector/source label to attach to each row (e.g., "travel_pass")

    Returns:
    - A pandas DataFrame with the original columns plus a `source` column
    """
    path = Path(csv_path)
    df = pd.read_csv(path)
    if question_col not in df.columns or answer_col not in df.columns:
        missing = {question_col, answer_col} - set(df.columns)
        raise ValueError(f"Q&A CSV missing required columns: {missing}")
    df = df.copy()
    df["source"] = source_label
    return df


def qa_dataframe_to_documents(
    df: pd.DataFrame,
    *,
    question_col: str = "question",
    answer_col: str = "answer",
    mode: str = "concat",
) -> tuple[list[str], list[dict]]:
    """Convert a Q&A dataframe into model-ingestable documents and aligned metadatas.

    Parameters:
    - df: dataframe produced by `load_qa_csv`
    - question_col/answer_col: column names containing the text
    - mode:
        - "concat": document is "Q: <question>\nA: <answer>"
        - "answer": document is just the answer text

    Returns:
    - (documents, metadatas) where `metadatas[i]` corresponds to `documents[i]`
    """
    docs: list[str] = []
    metas: list[dict] = []
    for _, row in df.iterrows():
        q = str(row[question_col])
        a = str(row[answer_col])
        if mode == "answer":
            doc = a
        else:
            doc = f"Q: {q}\nA: {a}"
        docs.append(doc)
        metas.append({
            "source": str(row.get("source", "qa")),
            "question": q,
        })
    return docs, metas
