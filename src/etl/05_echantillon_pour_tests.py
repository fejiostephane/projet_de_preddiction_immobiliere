"""
05_echantillon_pour_tests.py

Objectif
--------
Vous ne pouvez pas tester vos fonctions de consolidation sur les fichiers DVF
complets (plusieurs centaines de Mo à quelques Go par année) car chaque essai
prend trop de temps. Ce script résout ce problème de deux façons
complémentaires :

1. Il génère un ÉCHANTILLON représentatif de chaque fichier .txt (ex: 50 000
   lignes tirées de manière homogène, pas juste les 50 000 premières, pour
   éviter un biais géographique/temporel).
2. Il convertit les fichiers en Parquet (colonne compressée), ce qui divise
   par 5 à 10 le temps de lecture par rapport au CSV pipe-delimited, même
   pour le dataset complet.

Usage :
    - Pendant le développement -> travaillez sur les fichiers dans
      data/samples/ (rapides à charger).
    - Une fois vos fonctions validées -> relancez la même logique sur
      data/parquet/ (fichiers complets, mais bien plus rapides que les .txt).
"""

import os
from pathlib import Path

import pandas as pd


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
DATA_DIR = PROJET_RACINE / "data"

EXTRACT_DIR = DATA_DIR / "extracted" / "dvf"
SAMPLE_DIR = DATA_DIR / "samples" / "dvf"
PARQUET_DIR = DATA_DIR / "parquet" / "dvf"

SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

# Colonnes réellement utilisées en aval (cf. notebooks 03/04).
# Limiter les colonnes dès la lecture réduit fortement la mémoire utilisée.
COLONNES_UTILES = [
    "Date mutation",
    "Nature mutation",
    "Valeur fonciere",
    "Code postal",
    "Commune",
    "Code departement",
    "Code commune",
    "Type local",
    "Surface reelle bati",
    "Nombre pieces principales",
    "Surface terrain",
]

# Types explicites -> évite que pandas déduise des types coûteux (object partout)
DTYPES = {
    "Nature mutation": "category",
    "Commune": "category",
    "Code departement": "category",
    "Type local": "category",
}


def compter_lignes(fichier: Path) -> int:
    """Compte les lignes sans charger tout le fichier en mémoire."""
    with open(fichier, "r", encoding="latin-1", errors="ignore") as f:
        return sum(1 for _ in f) - 1  # -1 pour l'en-tête


def creer_echantillon(fichier: Path, n_lignes: int = 50_000) -> pd.DataFrame:
    """
    Construit un échantillon représentatif en lisant le fichier par blocs
    (chunksize) et en piochant une fraction régulière de chaque bloc,
    plutôt que de prendre uniquement les premières lignes du fichier
    (qui sont souvent triées par département -> échantillon biaisé).
    """
    total_lignes = compter_lignes(fichier)
    frac = min(1.0, n_lignes / max(total_lignes, 1))

    print(f"{fichier.name} : {total_lignes:,} lignes -> échantillon ~{frac:.2%}")

    morceaux = []
    lecteur = pd.read_csv(
        fichier,
        sep="|",
        usecols=lambda c: c in COLONNES_UTILES,
        dtype={k: v for k, v in DTYPES.items() if k in COLONNES_UTILES},
        chunksize=100_000,
        low_memory=False,
    )

    for bloc in lecteur:
        morceaux.append(bloc.sample(frac=frac, random_state=42))

    return pd.concat(morceaux, ignore_index=True)


def convertir_en_parquet(fichier: Path) -> None:
    """Convertit un .txt complet en .parquet une bonne fois pour toutes."""
    sortie = PARQUET_DIR / (fichier.stem + ".parquet")
    if sortie.exists():
        print(f"Déjà converti : {sortie.name}")
        return

    print(f"Conversion en Parquet : {fichier.name}")
    lecteur = pd.read_csv(
        fichier,
        sep="|",
        usecols=lambda c: c in COLONNES_UTILES,
        dtype={k: v for k, v in DTYPES.items() if k in COLONNES_UTILES},
        chunksize=200_000,
        low_memory=False,
    )

    morceaux = [bloc for bloc in lecteur]
    df = pd.concat(morceaux, ignore_index=True)
    df.to_parquet(sortie, index=False)
    print(f"  -> {sortie} ({len(df):,} lignes, {sortie.stat().st_size / 1e6:.1f} Mo)")


def main():
    fichiers = sorted(EXTRACT_DIR.glob("*.txt"))
    if not fichiers:
        print("Aucun fichier .txt trouvé dans", EXTRACT_DIR)
        return

    for fichier in fichiers:
        # 1. échantillon léger pour le développement
        echantillon = creer_echantillon(fichier, n_lignes=50_000)
        chemin_sample = SAMPLE_DIR / (fichier.stem + "_sample.parquet")
        echantillon.to_parquet(chemin_sample, index=False)
        print(f"  -> échantillon sauvegardé : {chemin_sample} ({len(echantillon):,} lignes)\n")

        # 2. version complète en Parquet pour le run final
        convertir_en_parquet(fichier)


if __name__ == "__main__":
    main()
