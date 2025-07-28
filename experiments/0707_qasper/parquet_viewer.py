# file: parquet_viewer.py
import os
from pathlib import Path
from typing import List

from fastapi.staticfiles import StaticFiles
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from datasets import load_dataset

PARQUET_PATH = os.getenv("PARQUET_VIEWER_FILE", "data/my_dataset.parquet")  # ← override with env-var
PARQUET_SPLIT = os.getenv("PARQUET_VIEWER_SPLIT", "train")  # ← override with env-var
print(PARQUET_PATH)
DF = load_dataset('parquet', data_files=PARQUET_PATH)[PARQUET_SPLIT]                   # one-time load; fast with pyarrow

app = FastAPI(title="Parquet Row Explorer")

def build_row_payload(idx: int) -> dict:
    """
    Extract required fields from the requested row and shape them
    exactly as the user wants.
    """
    if idx < 0 or idx >= len(DF):
        raise IndexError(f"Index {idx} out of range 0–{len(DF)-1}")

    row = DF[idx]

    # 'generations' and 'answers' are assumed to be lists of dicts already.
    payload = {
        "title": row["title"],
        "abstract": row["abstract"],
        "question": row["question"],
        'nlp_background': row['nlp_background'],
        'topic_background': row['topic_background'],
        'paper_read': row['paper_read'],
        'search_query': row['search_query'],
        "generations": [
            {
                "mode": g["mode"],
                "model": g["model"],
                "temperature": g["temperature"],
                "max_completion_tokens": g["max_completion_tokens"],
                "content": g["content"],            # list[str]
            }
            for g in row["generations"]
        ],
        "majority_type": row["majority_type"],
        "majority_answer": row["majority_answer"],
        "answers": [
            {
                "type": a["type"],
                "answer_unified": a["answer_unified"],
                "evidence": a["evidence"],
                "highlighted_evidence": a["highlighted_evidence"],
            }
            for a in row["answers"]
        ],
    }
    return payload

@app.get("/row/{idx}", summary="Get a single row by index")
def get_row(idx: int):
    """
    Returns the selected row as JSON, with only the requested fields.
    """
    try:
        return JSONResponse(build_row_payload(idx))
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
def index():
    return FileResponse("static/index.html")