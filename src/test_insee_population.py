import os
import requests

RAW_DIR = "../data/raw/insee"
os.makedirs(RAW_DIR, exist_ok=True)

url = "https://opendata.isere.fr/api/explore/v2.1/catalog/datasets/populations-legales-communes-et-arrondissements-municipaux-millesime-france/exports/csv?use_labels=true"

output_file = os.path.join(RAW_DIR, "population_communes.csv")

response = requests.get(url)

if response.status_code == 200:
    with open(output_file, "wb") as f:
        f.write(response.content)

    print(f"✅ Fichier téléchargé : {output_file}")
else:
    print("❌ Erreur :", response.status_code)