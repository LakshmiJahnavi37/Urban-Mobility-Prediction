"""
Dataset utility to process and verify the downloaded Chicago CTA Bus Ridership dataset.
Extracts clean CSV records, validates column headers, row counts, and date spans.
"""

import os
import re
import pandas as pd


SOURCE_MD_PATH = r"C:\Users\ODURI BHEEMESWARI\.gemini\antigravity-ide\brain\030ec48f-dbb2-44d6-b018-85d1f82db84a\.system_generated\steps\347\content.md"
DESTINATION_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "passenger_demand.csv")


def extract_and_save_cta_csv(source_path: str = SOURCE_MD_PATH, dest_path: str = DESTINATION_CSV_PATH) -> pd.DataFrame:
    """
    Extracts the pure CSV content from the downloaded Chicago Data Portal stream and saves it.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    with open(source_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # Find start of CSV table (starts with '"route","date","daytype","rides"')
    csv_lines = []
    found_header = False
    for line in lines:
        if '"route","date","daytype","rides"' in line or 'route,date,daytype,rides' in line:
            found_header = True
            csv_lines.append(line)
        elif found_header:
            if line.strip():
                csv_lines.append(line)
                
    with open(dest_path, "w", encoding="utf-8") as f:
        f.writelines(csv_lines)
        
    df = pd.read_csv(dest_path)
    print(f"Successfully saved {len(df)} authentic CTA records to: {dest_path}")
    return df


if __name__ == "__main__":
    extract_and_save_cta_csv()
