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
RAW_DIR = PROJET_RACINE / "data" / "raw" / "insee"
RAW_DIR.mkdir(parents=True, exist_ok=True)

print(f"Racine du projet détectée : {PROJET_RACINE}")

url = "https://static.data.gouv.fr/resources/niveau-de-vie-des-francais-par-commune/20171031-164656/FRANCE_COMMUNE_NIVEAU_DE_VIE-FIGARO.csv"

response = requests.get(url)

sortie = RAW_DIR / "revenus_communes.csv"
with open(sortie, "wb") as f:
    f.write(response.content)

print(f"✅ Téléchargement terminé : {sortie}")
