import os
import requests
import zipfile
import io

DATA_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
DATA_DIR = "data"

def download_and_extract_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    print(f"Downloading data from {DATA_URL}...")
    try:
        response = requests.get(DATA_URL)
        response.raise_for_status()
        
        print("Extracting data...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(DATA_DIR)
            
        print(f"Data successfully downloaded and extracted to {DATA_DIR}")
        
        # Verify files
        files = os.listdir(DATA_DIR)
        print("Files in data directory:", files)
        
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_and_extract_data()
