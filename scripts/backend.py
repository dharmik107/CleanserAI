import sys
import os
import pandas as pd
import io
import aiohttp
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from sqlalchemy import create_engine
from pydantic import BaseModel
import requests

# Ensure the scripts folder is in Python's path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from scripts.ai_agent import AIAgent  # Import AI Agent
from scripts.data_cleaning import DataCleaning  # Import Rule-Based Data Cleaning

app = FastAPI()

# Initialize AI agent and rule-based data cleaner
ai_agent = AIAgent()
cleaner = DataCleaning()

# ------------------------ CSV / Excel Cleaning Endpoint ------------------------

@app.post("/clean-data")
async def clean_data(file: UploadFile = File(...)):
    """Receives file from UI, cleans it using rule-based & AI methods, and returns cleaned JSON."""
    try:
        contents = await file.read()
        file_extension = file.filename.split(".")[-1]

        # Load file into Pandas DataFrame
        if file_extension == "csv":
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        elif file_extension == "xlsx":
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or Excel.")

        # Step 1: Rule-Based Cleaning
        df_cleaned = cleaner.clean_data(df)


        # Step 2: AI-Powered Cleaning
        df_ai_cleaned = ai_agent.process_data(df_cleaned)

        # Ensure AI output is a DataFrame, with error handling and fallback
        if isinstance(df_ai_cleaned, str):
            from io import StringIO
            import json
            try:
                df_ai_cleaned = pd.read_csv(StringIO(df_ai_cleaned))
            except Exception as csv_e:
                # Log the problematic string for debugging
                print("AI output not valid CSV. Output was:\n", df_ai_cleaned)
                # Try to parse as JSON
                try:
                    data = json.loads(df_ai_cleaned)
                    df_ai_cleaned = pd.DataFrame(data)
                except Exception as json_e:
                    raise HTTPException(status_code=500, detail=f"AI output could not be parsed as CSV or JSON. CSV error: {csv_e}, JSON error: {json_e}. AI output: {df_ai_cleaned}")

        return {"cleaned_data": df_ai_cleaned.to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# ------------------------ Run Server ------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
