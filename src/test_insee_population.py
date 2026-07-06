from pathlib import Path

import requests


def trouver_racine_projet(depart: Path, marqueurs=("data", ".git")) -> Path:
    """
    Remonte l'arborescence depuis le dossier du script jusqu'à trouver un
    dossier contenant un des marqueurs (ex: "data/", ou ".git/").
    Ça évite de dépendre du nombre exact de sous-dossiers entre le script
    et la racine du projet (src/, src/etl/, src/ingestion/, etc. peuvent
    tous fonctionner sans rien changer au code).
    """
    courant = depart.resolve()
    for _ in range(6):  # on ne remonte pas indéfiniment
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

# ATTENTION : l'ancienne URL (opendata.isere.fr) ne contenait QUE les
# communes du département de l'Isère (38), malgré son nom trompeur.
# Voici la vraie source nationale (même producteur INSEE, mêmes colonnes),
# hébergée sur le portail Opendatasoft public :
url = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "demographyref-france-pop-legale-commune-arrondissement-municipal-millesime/"
    "exports/csv?lang=fr&timezone=Europe%2FParis&use_labels=true&delimiter=%3B"
)

output_file = RAW_DIR / "population_communes.csv"

response = requests.get(url)

if response.status_code == 200:
    with open(output_file, "wb") as f:
        f.write(response.content)

    print(f"✅ Fichier téléchargé : {output_file}")
    print(f"   Taille : {output_file.stat().st_size / 1e6:.1f} Mo")
else:
    print("❌ Erreur :", response.status_code)
