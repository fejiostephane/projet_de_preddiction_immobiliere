"""
06_modelisation_ml.py

Partie Machine Learning du projet DVF.

Hypothèse de départ : le notebook 04_consolidation_dvf a produit un
df_final contenant, entre autres, les colonnes suivantes (à adapter selon
votre notebook) :

    Date mutation, Valeur fonciere, Type local, Surface reelle bati,
    Nombre pieces principales, Surface terrain, Code departement,
    latitude, longitude, population_geo / population_totale,
    revenu_median, annee, mois, trimestre, prix_m2

et qu'il a été sauvegardé en Parquet (bien plus rapide à recharger qu'un
gros CSV) via :

    df_final.to_parquet("../../data/processed/dvf_final.parquet", index=False)

Problème métier retenu : prédire le prix au m² (prix_m2) d'un bien
(maison ou appartement) à partir de ses caractéristiques et de son
environnement socio-économique (commune, revenu médian, population).
C'est un problème de régression supervisée.

Le script :
1. Charge le dataset (ou l'échantillon, cf. 05_echantillon_pour_tests.py,
   pendant la phase de mise au point).
2. Prépare les features (encodage, découpage train/test temporel).
3. Entraîne plusieurs modèles et les compare (baseline -> avancé).
4. Sauvegarde le meilleur modèle et un tableau comparatif des métriques.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------------------------------------------------
# 0. Configuration
# -----------------------------------------------------------------------


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
DATA_PATH = PROJET_RACINE / "data" / "processed" / "dvf_final.parquet"

OUTPUT_DIR = PROJET_RACINE / "data" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_DIR = OUTPUT_DIR / "graphiques"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "prix_m2"

FEATURES_NUM = [
    "Surface reelle bati",
    "Nombre pieces principales",
    "Surface terrain",
    "latitude",
    "longitude",
    "population_geo",
    "revenu_median",
    "annee",
    "mois",
]

FEATURES_CAT = [
    "Type local",
    "Code departement",
]

RANDOM_STATE = 42


# -----------------------------------------------------------------------
# 1. Chargement et préparation
# -----------------------------------------------------------------------

def charger_donnees(path: Path) -> pd.DataFrame:
    print(f"Chargement de {path} ...")
    df = pd.read_parquet(path)
    print(f"  -> {len(df):,} lignes, {df.shape[1]} colonnes")
    return df


def preparer_features(df: pd.DataFrame):
    """
    Ne garde que les colonnes utiles au modèle, supprime les lignes
    incomplètes sur ces colonnes, et applique une transformation log au
    prix au m2 : la distribution des prix est très asymétrique
    (cf. vos histogrammes du notebook 03), donc log1p stabilise la
    variance et améliore nettement les régressions linéaires.
    """
    colonnes = FEATURES_NUM + FEATURES_CAT + [TARGET]
    colonnes = [c for c in colonnes if c in df.columns]
    df = df[colonnes].dropna().copy()

    df["Code departement"] = df["Code departement"].astype(str)

    X = df[[c for c in FEATURES_NUM if c in df.columns] +
           [c for c in FEATURES_CAT if c in df.columns]]
    y_log = np.log1p(df[TARGET])

    return X, y_log, df


def decouper_train_test(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
    """
    Découpage TEMPOREL plutôt qu'aléatoire : on entraîne sur les années
    passées et on teste sur la dernière année disponible. C'est plus
    honnête pour un cas d'usage réel (prédire des prix futurs) et évite
    une fuite d'information entre biens vendus la même semaine dans le
    même quartier.
    """
    if "annee" in df.columns and df["annee"].nunique() > 1:
        derniere_annee = df["annee"].max()
        mask_test = df["annee"] == derniere_annee
        print(f"Découpage temporel : test = année {derniere_annee}")
    else:
        # Fallback si une seule année est disponible (ex: échantillon de test)
        from sklearn.model_selection import train_test_split
        idx_train, idx_test = train_test_split(
            df.index, test_size=0.2, random_state=RANDOM_STATE
        )
        mask_test = df.index.isin(idx_test)
        print("Une seule année disponible -> découpage aléatoire 80/20")

    X_train, X_test = X[~mask_test], X[mask_test]
    y_train, y_test = y[~mask_test], y[mask_test]

    print(f"  Train : {len(X_train):,} lignes | Test : {len(X_test):,} lignes")
    return X_train, X_test, y_train, y_test


# -----------------------------------------------------------------------
# 2. Construction du préprocesseur (commun à tous les modèles)
# -----------------------------------------------------------------------

def construire_preprocesseur(features_num, features_cat):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), features_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
        ]
    )


# -----------------------------------------------------------------------
# 3. Modèles à comparer
#    - Baselines : Régression linéaire, Ridge (supervisé classique)
#    - Modèle avancé : Gradient Boosting (capte les non-linéarités et
#      interactions entre surface / localisation / revenu, ce qu'une
#      régression linéaire ne peut pas faire)
# -----------------------------------------------------------------------

def construire_modeles():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
    }


def evaluer(y_true_log, y_pred_log):
    """
    On repasse en euros (expm1) avant de calculer les métriques,
    car MAE/RMSE en "log-euros" ne veulent rien dire pour un utilisateur
    final. R2 reste calculé sur l'échelle log (plus stable).
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    return {
        "MAE (€/m2)": mean_absolute_error(y_true, y_pred),
        "RMSE (€/m2)": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2 (log)": r2_score(y_true_log, y_pred_log),
    }


# -----------------------------------------------------------------------
# 3bis. Visualisations
# -----------------------------------------------------------------------

def graphique_comparaison_modeles(tableau: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, colonne, titre, couleur in zip(
        axes,
        ["R2 (log)", "MAE (€/m2)", "RMSE (€/m2)"],
        ["R² (échelle log)", "MAE (€/m2)", "RMSE (€/m2)"],
        ["#4C72B0", "#DD8452", "#C44E52"],
    ):
        data = tableau.sort_values(colonne, ascending=(colonne != "R2 (log)"))
        ax.barh(data["modele"], data[colonne], color=couleur)
        ax.set_title(titre)
        ax.invert_yaxis()
        for i, v in enumerate(data[colonne]):
            ax.text(v, i, f" {v:,.0f}" if v > 100 else f" {v:.3f}", va="center")

    fig.suptitle("Comparaison des modèles", fontsize=14)
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "01_comparaison_modeles.png", dpi=150)
    plt.close(fig)
    print(f"  -> {GRAPH_DIR / '01_comparaison_modeles.png'}")


def graphique_predictions_vs_reel(y_test_log, y_pred_log, nom_modele: str):
    y_test = np.expm1(y_test_log)
    y_pred = np.expm1(y_pred_log)

    # On limite l'affichage au 99e percentile pour que les rares valeurs
    # extrêmes n'écrasent pas le nuage de points principal
    limite = np.percentile(y_test, 99)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.15, s=10, color="#4C72B0")
    ax.plot([0, limite], [0, limite], color="red", linestyle="--", label="Prédiction parfaite")
    ax.set_xlim(0, limite)
    ax.set_ylim(0, limite)
    ax.set_xlabel("Prix au m² réel (€)")
    ax.set_ylabel("Prix au m² prédit (€)")
    ax.set_title(f"Prédictions vs réalité — {nom_modele}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "02_predictions_vs_reel.png", dpi=150)
    plt.close(fig)
    print(f"  -> {GRAPH_DIR / '02_predictions_vs_reel.png'}")


def graphique_residus(y_test_log, y_pred_log, nom_modele: str):
    residus = y_test_log - y_pred_log

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(residus, bins=60, color="#55A868")
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_title("Distribution des résidus (échelle log)")
    axes[0].set_xlabel("Erreur (log réel - log prédit)")

    axes[1].scatter(y_pred_log, residus, alpha=0.15, s=10, color="#55A868")
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title("Résidus vs valeurs prédites")
    axes[1].set_xlabel("log(prix_m2) prédit")
    axes[1].set_ylabel("Résidu")

    fig.suptitle(f"Analyse des erreurs — {nom_modele}", fontsize=14)
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "03_residus.png", dpi=150)
    plt.close(fig)
    print(f"  -> {GRAPH_DIR / '03_residus.png'}")


def graphique_importance_variables(pipeline: Pipeline, nom_modele: str):
    modele = pipeline.named_steps["model"]
    if not hasattr(modele, "feature_importances_"):
        print(f"  (pas d'importance de variables disponible pour {nom_modele})")
        return

    noms_features = pipeline.named_steps["prep"].get_feature_names_out()
    importances = modele.feature_importances_

    top = pd.Series(importances, index=noms_features).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.index[::-1], top.values[::-1], color="#8172B2")
    ax.set_title(f"Importance des variables — {nom_modele} (top 15)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "04_importance_variables.png", dpi=150)
    plt.close(fig)
    print(f"  -> {GRAPH_DIR / '04_importance_variables.png'}")


def graphique_distribution_et_evolution(df_prep: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    limite = df_prep["prix_m2"].quantile(0.99)
    axes[0].hist(
        df_prep[df_prep["prix_m2"] <= limite]["prix_m2"],
        bins=60,
        color="#4C72B0",
    )
    axes[0].set_title("Distribution du prix au m² (hors 1% extrêmes)")
    axes[0].set_xlabel("Prix au m² (€)")

    if "annee" in df_prep.columns:
        evolution = df_prep.groupby("annee")["prix_m2"].median()
        axes[1].plot(evolution.index, evolution.values, marker="o", color="#C44E52")
        axes[1].set_title("Prix médian au m² par année")
        axes[1].set_xlabel("Année")
        axes[1].set_ylabel("Prix médian au m² (€)")
        axes[1].set_xticks(evolution.index)

    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "05_distribution_et_evolution.png", dpi=150)
    plt.close(fig)
    print(f"  -> {GRAPH_DIR / '05_distribution_et_evolution.png'}")


# -----------------------------------------------------------------------
# 4. Pipeline principal
# -----------------------------------------------------------------------

def main():
    df_brut = charger_donnees(DATA_PATH)
    X, y, df_prep = preparer_features(df_brut)
    X_train, X_test, y_train, y_test = decouper_train_test(df_prep, X, y)

    features_num = [c for c in FEATURES_NUM if c in X.columns]
    features_cat = [c for c in FEATURES_CAT if c in X.columns]

    resultats = []
    meilleurs = {"nom": None, "score": -np.inf, "pipeline": None, "y_pred": None}

    for nom, modele in construire_modeles().items():
        print(f"\nEntraînement : {nom}")
        pipeline = Pipeline(
            steps=[
                ("prep", construire_preprocesseur(features_num, features_cat)),
                ("model", modele),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metriques = evaluer(y_test, y_pred)
        metriques["modele"] = nom
        resultats.append(metriques)

        print(f"  MAE  = {metriques['MAE (€/m2)']:.0f} €/m2")
        print(f"  RMSE = {metriques['RMSE (€/m2)']:.0f} €/m2")
        print(f"  R2   = {metriques['R2 (log)']:.3f}")

        if metriques["R2 (log)"] > meilleurs["score"]:
            meilleurs = {
                "nom": nom,
                "score": metriques["R2 (log)"],
                "pipeline": pipeline,
                "y_pred": y_pred,
            }

    # Tableau comparatif
    tableau = pd.DataFrame(resultats)[
        ["modele", "MAE (€/m2)", "RMSE (€/m2)", "R2 (log)"]
    ].sort_values("R2 (log)", ascending=False)

    print("\n=== Comparaison des modèles ===")
    print(tableau.to_string(index=False))

    tableau.to_csv(OUTPUT_DIR / "comparaison_modeles.csv", index=False)

    # Sauvegarde du meilleur modèle (pipeline complet : preprocessing + modèle)
    chemin_modele = OUTPUT_DIR / "meilleur_modele_prix_m2.joblib"
    joblib.dump(meilleurs["pipeline"], chemin_modele)

    with open(OUTPUT_DIR / "meta_modele.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "modele_retenu": meilleurs["nom"],
                "r2_log": meilleurs["score"],
                "features_num": features_num,
                "features_cat": features_cat,
                "target": TARGET,
                "transformation_target": "log1p",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nModèle retenu : {meilleurs['nom']} (R2 log = {meilleurs['score']:.3f})")
    print(f"Sauvegardé dans : {chemin_modele}")

    # Visualisations
    print("\nGénération des visualisations...")
    graphique_comparaison_modeles(tableau)
    graphique_predictions_vs_reel(y_test, meilleurs["y_pred"], meilleurs["nom"])
    graphique_residus(y_test, meilleurs["y_pred"], meilleurs["nom"])
    graphique_importance_variables(meilleurs["pipeline"], meilleurs["nom"])
    graphique_distribution_et_evolution(df_prep)
    print(f"\nGraphiques sauvegardés dans : {GRAPH_DIR}")


if __name__ == "__main__":
    main()
