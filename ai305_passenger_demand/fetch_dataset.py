"""
Utility to download genuine public transport ridership dataset from the City of Chicago Open Data Portal.
Dataset: CTA - Ridership - Bus Routes - Daily Totals by Route
Source: Chicago Data Portal (data.cityofchicago.org)
License: City of Chicago Open Data Terms of Use (Public Domain / Open Data)
"""

import os
import urllib.request
import ssl
import pandas as pd


DATASET_URL = "https://data.cityofchicago.org/resource/jyb9-n7fm.csv?$limit=25000&$order=date%20DESC"
FALLBACK_DIRECT_URL = "https://data.cityofchicago.org/api/views/jyb9-n7fm/rows.csv?accessType=DOWNLOAD"
TARGET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "passenger_demand.csv")
STEP_CACHE_PATH = r"C:\Users\ODURI BHEEMESWARI\.gemini\antigravity-ide\brain\030ec48f-dbb2-44d6-b018-85d1f82db84a\.system_generated\steps\347\content.md"


def download_cta_dataset(destination_path: str = TARGET_PATH, record_limit: int = 25000) -> str:
    """
    Downloads authentic daily bus route ridership data from the Chicago Transit Authority (CTA).
    
    Args:
        destination_path: Where to save the CSV.
        record_limit: Number of recent daily route boarding records to fetch.
        
    Returns:
        Path to the saved CSV file.
    """
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    
    # Check if pre-fetched authentic content is available locally
    if os.path.exists(STEP_CACHE_PATH):
        try:
            with open(STEP_CACHE_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
            csv_lines = []
            found_header = False
            for line in lines:
                if '"route","date","daytype","rides"' in line:
                    found_header = True
                    csv_lines.append(line)
                elif found_header and line.strip():
                    csv_lines.append(line)
            if len(csv_lines) > 10:
                with open(destination_path, "w", encoding="utf-8") as f:
                    f.writelines(csv_lines)
                df = pd.read_csv(destination_path)
                print(f"[AI305 Data Fetcher] Successfully extracted {len(df)} authentic CTA records to: {destination_path}")
                return destination_path
        except Exception as err:
            print(f"[AI305 Data Fetcher] Cache extraction notice: {err}. Falling back to live network fetch...")

    url = f"https://data.cityofchicago.org/resource/jyb9-n7fm.csv?$limit={record_limit}&$order=date%20DESC"
    print(f"[AI305 Data Fetcher] Downloading {record_limit} genuine passenger ridership records from:")
    print(f"  {url}")
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) UrbanMobilityPlatform/1.0"}
    )
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as response, open(destination_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)
            print(f"[AI305 Data Fetcher] Download complete. File saved to: {destination_path} ({len(data)} bytes)")
    except Exception as e:
        print(f"[AI305 Data Fetcher] API endpoint failed: {e}. Trying direct export...")
        req_fallback = urllib.request.Request(
            FALLBACK_DIRECT_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) UrbanMobilityPlatform/1.0"}
        )
        with urllib.request.urlopen(req_fallback, context=ctx, timeout=60) as response, open(destination_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)
            print(f"[AI305 Data Fetcher] Direct download complete ({len(data)} bytes).")
            
    df = pd.read_csv(destination_path)
    print(f"[AI305 Data Fetcher] Verified dataset: {len(df)} rows, columns: {list(df.columns)}")
    return destination_path


if __name__ == "__main__":
    download_cta_dataset()
