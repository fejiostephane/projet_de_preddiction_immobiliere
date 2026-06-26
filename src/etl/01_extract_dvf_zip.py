import os
import zipfile

RAW_DIR = "../../data/raw/dvf"
EXTRACT_DIR = "../../data/extracted/dvf"

os.makedirs(EXTRACT_DIR, exist_ok=True)

for file_name in os.listdir(RAW_DIR):
    if file_name.endswith(".zip"):
        zip_path = os.path.join(RAW_DIR, file_name)

        print(f"Décompression : {file_name}")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

        print(f"OK : {file_name}")