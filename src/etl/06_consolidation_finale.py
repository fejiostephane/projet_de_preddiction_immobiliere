"""
06_consolidation_finale.py

Combine tous les fichiers Parquet annuels générés par
05_echantillon_pour_tests.py en UN SEUL dataset, applique le même
nettoyage que le notebook 03/04, puis enrichit avec :
    - les coordonnées géographiques des communes (communes_france.json)
    - le revenu médian par commune (revenus_communes.csv)
    - la population par commune (population_communes.csv)

Résultat : data/processed/dvf_final.parquet
           (c'est ce fichier que lit 06_modelisation_ml.py)

Usage :
    MODE = "sample" -> travaille sur data/samples/dvf/*.parquet (rapide,
                        pour valider que tout le script tourne sans erreur)
    MODE = "full"   -> travaille sur data/parquet/dvf/*.parquet (le vrai
                        dataset complet, une fois que MODE="sample" a
                        tourné sans erreur)
"""

from pathlib import Path

import pandas as pd

# -----------------------------------------------------------------------
# 0. Configuration
# -----------------------------------------------------------------------

MODE = "sample"  # "sample" pendant les tests, "full" pour le run final


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

SAMPLE_DIR = DATA_DIR / "samples" / "dvf"
PARQUET_DIR = DATA_DIR / "parquet" / "dvf"

GEO_PATH = DATA_DIR / "raw" / "geo" / "communes_france.json"
REVENUS_PATH = DATA_DIR / "raw" / "insee" / "revenus_communes.csv"
POPULATION_PATH = DATA_DIR / "raw" / "insee" / "population_communes.csv"

OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "dvf_final.parquet"
OUTPUT_CSV_PATH = OUTPUT_DIR / "dvf_final_propre.csv"

print(f"Racine du projet détectée : {PROJET_RACINE}")


def verifier_fichiers_reference():
    manquants = [p for p in [GEO_PATH, REVENUS_PATH, POPULATION_PATH] if not p.exists()]
    if manquants:
        print("\n⚠️  Fichiers de référence manquants :")
        for p in manquants:
            print(f"   - {p}")
        print(
            "\nLancez d'abord (dans cet ordre) :\n"
            "  python referenciel_des_comunes.py\n"
            "  python revenu_comunes_data.py\n"
            "  python test_insee_population.py\n"
        )
        raise FileNotFoundError("Fichiers de référence manquants, voir ci-dessus.")


# -----------------------------------------------------------------------
# 1. Concaténation des fichiers annuels
# -----------------------------------------------------------------------

def charger_et_concatener(mode: str) -> pd.DataFrame:
    dossier = SAMPLE_DIR if mode == "sample" else PARQUET_DIR
    suffixe = "_sample.parquet" if mode == "sample" else ".parquet"

    fichiers = sorted(dossier.glob(f"*{suffixe}"))
    if not fichiers:
        raise FileNotFoundError(
            f"Aucun fichier trouvé dans {dossier}. "
            f"Avez-vous bien lancé 05_echantillon_pour_tests.py ?"
        )

    print(f"Mode = {mode} | {len(fichiers)} fichier(s) trouvé(s) dans {dossier}")

    morceaux = []
    for fichier in fichiers:
        df_annee = pd.read_parquet(fichier)
        print(f"  {fichier.name} : {len(df_annee):,} lignes")
        morceaux.append(df_annee)

    df = pd.concat(morceaux, ignore_index=True)
    print(f"Total combiné : {len(df):,} lignes\n")
    return df


# -----------------------------------------------------------------------
# 2. Nettoyage (reprend la logique des notebooks 03 / 04)
# -----------------------------------------------------------------------

def nettoyer(df: pd.DataFrame) -> pd.DataFrame:
    df = df[
        (df["Nature mutation"] == "Vente")
        & (df["Type local"].isin(["Maison", "Appartement"]))
    ].copy()

    df["Date mutation"] = pd.to_datetime(
        df["Date mutation"], dayfirst=True, errors="coerce"
    )

    if df["Valeur fonciere"].dtype == object:
        df["Valeur fonciere"] = (
            df["Valeur fonciere"].str.replace(",", ".", regex=False).astype(float)
        )

    df = df.dropna(
        subset=[
            "Valeur fonciere",
            "Code postal",
            "Surface reelle bati",
            "Nombre pieces principales",
        ]
    )

    # Filtres de cohérence identiques au notebook 04
    df = df[df["Valeur fonciere"] > 1000]
    df = df[df["Valeur fonciere"] < 100_000_000]
    df = df[df["Surface reelle bati"] > 5]

    df = df.drop_duplicates()

    df["prix_m2"] = df["Valeur fonciere"] / df["Surface reelle bati"]

    # On retire les prix/m2 aberrants (>99e percentile), comme dans le notebook 03
    seuil_99 = df["prix_m2"].quantile(0.99)
    df = df[df["prix_m2"] <= seuil_99]

    df["annee"] = df["Date mutation"].dt.year
    df["mois"] = df["Date mutation"].dt.month
    df["trimestre"] = df["Date mutation"].dt.quarter

    print(f"Après nettoyage : {len(df):,} lignes\n")
    return df


# -----------------------------------------------------------------------
# 3. Code commune harmonisé (Paris/Lyon/Marseille -> code global)
# -----------------------------------------------------------------------

def corriger_code_commune(row) -> str:
    dep = str(row["Code departement"]).zfill(2)
    code = str(row["Code commune"])

    if dep == "75":
        return "75056"
    if dep == "13" and code.zfill(3) in [str(i).zfill(3) for i in range(201, 217)]:
        return "13055"
    if dep == "69" and code.zfill(3) in [str(i).zfill(3) for i in range(381, 390)]:
        return "69123"
    if len(dep) == 3:
        return dep + code.zfill(2)
    return dep + code.zfill(3)


# -----------------------------------------------------------------------
# 4. Enrichissement géo / revenu / population
# -----------------------------------------------------------------------

def enrichir(df: pd.DataFrame) -> pd.DataFrame:
    df["code_commune_geo"] = df.apply(corriger_code_commune, axis=1)

    # --- Géo (communes) ---
    communes = pd.read_json(GEO_PATH)
    communes["longitude"] = communes["centre"].apply(
        lambda x: x["coordinates"][0] if isinstance(x, dict) else None
    )
    communes["latitude"] = communes["centre"].apply(
        lambda x: x["coordinates"][1] if isinstance(x, dict) else None
    )
    communes = communes.drop(columns=["centre"]).rename(
        columns={
            "code": "code_commune_geo",
            "nom": "nom_commune_geo",
            "population": "population_geo",
        }
    )

    df = df.merge(communes, on="code_commune_geo", how="left")
    df = df.dropna(subset=["latitude", "longitude", "population_geo"])
    print(f"Après jointure géo : {len(df):,} lignes")

    # --- Revenu médian ---
    revenus = pd.read_csv(REVENUS_PATH, sep=";")
    if revenus["MED14"].dtype == object:
        revenus["MED14"] = revenus["MED14"].str.replace(",", ".", regex=False)
    revenus["MED14"] = pd.to_numeric(revenus["MED14"], errors="coerce")
    revenus = revenus.rename(
        columns={"CODGEO": "code_commune_geo", "MED14": "revenu_median"}
    )

    df = df.merge(
        revenus[["code_commune_geo", "revenu_median"]],
        on="code_commune_geo",
        how="left",
    )
    df = df.dropna(subset=["revenu_median"])
    print(f"Après jointure revenu : {len(df):,} lignes")

    # --- Population (recensement) ---
    population = pd.read_csv(POPULATION_PATH, sep=";")

    colonne_code = "Code Officiel Commune / Arrondissement Municipal"
    if colonne_code not in population.columns:
        print("\n⚠️  Colonne code commune introuvable dans population_communes.csv")
        print("Colonnes disponibles :", list(population.columns))
        raise KeyError(colonne_code)

    population_clean = population[
        [colonne_code, "Population municipale", "Population totale"]
    ].rename(
        columns={
            colonne_code: "code_commune_geo",
            "Population municipale": "population_municipale",
            "Population totale": "population_totale",
        }
    )

    # Nettoyage du code : suppression des espaces, forçage en texte 5 caractères
    population_clean["code_commune_geo"] = (
        population_clean["code_commune_geo"]
        .astype(str)
        .str.strip()
        .str.zfill(5)
    )
    df["code_commune_geo"] = df["code_commune_geo"].astype(str).str.strip().str.zfill(5)

    # --- Diagnostic avant jointure : pourquoi ça matche ou pas ---
    codes_df = set(df["code_commune_geo"])
    codes_pop = set(population_clean["code_commune_geo"])
    intersection = codes_df & codes_pop

    print(f"\nDiagnostic jointure population :")
    print(f"  Codes uniques côté DVF        : {len(codes_df)}")
    print(f"  Codes uniques côté population : {len(codes_pop)}")
    print(f"  Codes en commun               : {len(intersection)}")
    print(f"  Exemples DVF non trouvés      : {list(codes_df - codes_pop)[:5]}")
    print(f"  Exemples population (échant.) : {list(codes_pop)[:5]}")

    df = df.merge(population_clean, on="code_commune_geo", how="left")
    print(f"Après jointure population : {len(df):,} lignes")
    print(
        f"  -> population_municipale manquante : "
        f"{df['population_municipale'].isna().sum():,} lignes "
        f"({df['population_municipale'].isna().mean() * 100:.1f} %)\n"
    )

    return df


# -----------------------------------------------------------------------
# 5. Nettoyage final pour export CSV lisible (doublons, types, arrondis)
# -----------------------------------------------------------------------

def nettoyer_pour_export_csv(df: pd.DataFrame) -> pd.DataFrame:
    colonnes_a_retirer = [
        "Nature mutation",       # constante ("Vente" partout)
        "codeDepartement",       # doublon exact de "Code departement"
        "Code commune",          # redondant avec code_commune_geo
        "Commune",               # redondant avec nom_commune_geo (mieux formaté)
    ]
    df = df.drop(columns=[c for c in colonnes_a_retirer if c in df.columns])

    # Si la jointure population a marché, on garde population_totale et on
    # retire population_geo (moins fiable) ; sinon on garde population_geo
    # comme repli.
    if "population_totale" in df.columns and df["population_totale"].notna().mean() > 0.5:
        df = df.drop(columns=["population_geo"], errors="ignore")
        df = df.drop(columns=["population_municipale"], errors="ignore")
    else:
        df = df.drop(columns=["population_totale", "population_municipale"], errors="ignore")

    df = df.rename(columns={
        "Date mutation": "date_mutation",
        "Valeur fonciere": "valeur_fonciere_eur",
        "Code postal": "code_postal",
        "Code departement": "departement",
        "Type local": "type_bien",
        "Surface reelle bati": "surface_habitable_m2",
        "Nombre pieces principales": "nombre_pieces",
        "Surface terrain": "surface_terrain_m2",
        "code_commune_geo": "code_commune_insee",
        "nom_commune_geo": "commune",
        "codeRegion": "code_region",
        "population_geo": "population",
        "population_totale": "population",
        "revenu_median": "revenu_median_eur",
    })

    if "code_postal" in df.columns:
        df["code_postal"] = df["code_postal"].astype(int)
    if "code_region" in df.columns:
        df["code_region"] = df["code_region"].astype("Int64")
    if "population" in df.columns:
        df["population"] = df["population"].round(0).astype("Int64")
    if "valeur_fonciere_eur" in df.columns:
        df["valeur_fonciere_eur"] = df["valeur_fonciere_eur"].round(0).astype(int)
    if "surface_habitable_m2" in df.columns:
        df["surface_habitable_m2"] = df["surface_habitable_m2"].round(1)
    if "surface_terrain_m2" in df.columns:
        df["surface_terrain_m2"] = df["surface_terrain_m2"].round(1)
    if "prix_m2" in df.columns:
        df["prix_m2"] = df["prix_m2"].round(2)
    if "revenu_median_eur" in df.columns:
        df["revenu_median_eur"] = df["revenu_median_eur"].round(0).astype("Int64")
    if "latitude" in df.columns:
        df["latitude"] = df["latitude"].round(5)
    if "longitude" in df.columns:
        df["longitude"] = df["longitude"].round(5)
    if "nombre_pieces" in df.columns:
        df["nombre_pieces"] = df["nombre_pieces"].astype(int)

    ordre_souhaite = [
        "date_mutation", "annee", "mois", "trimestre",
        "type_bien", "valeur_fonciere_eur", "surface_habitable_m2",
        "surface_terrain_m2", "nombre_pieces", "prix_m2",
        "code_postal", "commune", "code_commune_insee", "departement",
        "population", "revenu_median_eur", "latitude", "longitude",
    ]
    colonnes_finales = [c for c in ordre_souhaite if c in df.columns]
    colonnes_restantes = [c for c in df.columns if c not in colonnes_finales]
    df = df[colonnes_finales + colonnes_restantes]

    if "date_mutation" in df.columns:
        df = df.sort_values("date_mutation").reset_index(drop=True)

    return df


# -----------------------------------------------------------------------
# 6. Pipeline principal
# -----------------------------------------------------------------------

def main():
    verifier_fichiers_reference()
    df = charger_et_concatener(MODE)
    df = nettoyer(df)
    df = enrichir(df)

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Dataset final sauvegardé (Parquet) : {OUTPUT_PATH}")
    print(f"  -> {len(df):,} lignes, {df.shape[1]} colonnes")
    print(f"\nColonnes disponibles :\n{list(df.columns)}")

    # Export CSV propre et lisible (mêmes données, colonnes nettoyées)
    df_propre = nettoyer_pour_export_csv(df)
    df_propre.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\nDataset final sauvegardé (CSV lisible) : {OUTPUT_CSV_PATH}")
    print(f"  -> {len(df_propre):,} lignes, {df_propre.shape[1]} colonnes")
    print(f"\nColonnes CSV finales :\n{list(df_propre.columns)}")


if __name__ == "__main__":
    main()
