import os
import pandas as pd
import requests
from sqlalchemy import create_engine

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")

class DataIngestion:
    def __init__(self, db_url=None):
        self.engine = create_engine(db_url)
        
    def load_csv(self, file_name):
        """Loads a CSV file into a DataFrame."""
        file_path = os.path.join(DATA_DIR, file_name)
        try:
            df = pd.read_csv(file_path)
            print(f"✅ CSV Loaded Successfully: {file_path}")
            return df
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return None

    def load_excel(self, file_name, sheet_name=0):
        """Loads an Excel file into a DataFrame."""
        file_path = os.path.join(DATA_DIR, file_name)
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            print(f"✅ Excel Loaded Successfully: {file_path}")
            return df
        except Exception as e:
            print(f"❌ Error loading Excel: {e}")
            return None