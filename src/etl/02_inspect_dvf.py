import pandas as pd
import os

EXTRACT_DIR = "../../data/extracted/dvf"

files = [f for f in os.listdir(EXTRACT_DIR) if f.endswith(".txt")]

print("Fichiers trouvés :")
print(files)

file_path = os.path.join(EXTRACT_DIR, files[0])

print(f"\nAnalyse de : {files[0]}")

df = pd.read_csv(
    file_path,
    sep="|",
    nrows=5,
    low_memory=False
)

print("\nColonnes :")
print(df.columns.tolist())

print("\nAperçu :")
print(df.head())

print("\nNombre de colonnes :", len(df.columns))