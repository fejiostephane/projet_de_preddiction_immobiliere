import json
from pathlib import Path

import requests


def trouver_racine_projet(depart: Path, marqueurs=("data", ".git")) -> Path:
    courant = depart.resolve()
    for _ in range(6):
        if any((courant / m).exists() for m in marqueurs):
            return courant
        courant = courant.parent
    raise RuntimeError(
        f"Racine du projet introuvable en remontant depuis {depart}. "
        f"Vérifiez qu'un dossier 'data/' existe bien quelque part au-dessus."
    )


PROJET_RACINE = trouver_racine_projet(Path(__file__).parent)
RAW_DIR = PROJET_RACINE / "data" / "raw" / "geo"
RAW_DIR.mkdir(parents=True, exist_ok=True)

print(f"Racine du projet détectée : {PROJET_RACINE}")

url = "https://geo.api.gouv.fr/communes?fields=nom,code,codeDepartement,codeRegion,population,centre&format=json"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    sortie = RAW_DIR / "communes_france.json"
    with open(sortie, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Fichier téléchargé : {sortie}")
    print("Nombre de communes :", len(data))
else:
    print("❌ Erreur :", response.status_code)
