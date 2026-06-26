import os
import requests
import json

RAW_DIR = "../data/raw/geo"
os.makedirs(RAW_DIR, exist_ok=True)

url = "https://geo.api.gouv.fr/communes?fields=nom,code,codeDepartement,codeRegion,population,centre&format=json"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    with open("../data/raw/geo/communes_france.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("✅ communes_france.json téléchargé")
    print("Nombre de communes :", len(data))
else:
    print("❌ Erreur :", response.status_code)