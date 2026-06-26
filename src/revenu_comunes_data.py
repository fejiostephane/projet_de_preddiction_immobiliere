import requests
import os

RAW_DIR = "../data/raw/insee"
os.makedirs(RAW_DIR, exist_ok=True)

url = "https://static.data.gouv.fr/resources/niveau-de-vie-des-francais-par-commune/20171031-164656/FRANCE_COMMUNE_NIVEAU_DE_VIE-FIGARO.csv"

response = requests.get(url)

with open(
    "../data/raw/insee/revenus_communes.csv",
    "wb"
) as f:
    f.write(response.content)

print("Téléchargement terminé")