"""
app_streamlit.py

Interface de démonstration pour le modèle de prédiction du prix
immobilier au m², entraîné par 06_modelisation_ml.py (renommé 07_ chez
vous).

Lancement :
    streamlit run app_streamlit.py
"""

from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------
# 0. Localisation des fichiers (même logique que les autres scripts)
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
MODELS_DIR = PROJET_RACINE / "data" / "models"
GRAPH_DIR = MODELS_DIR / "graphiques"

MODELE_PATH = MODELS_DIR / "meilleur_modele_prix_m2.joblib"
META_PATH = MODELS_DIR / "meta_modele.json"
COMPARAISON_PATH = MODELS_DIR / "comparaison_modeles.csv"
DATASET_PATH = PROJET_RACINE / "data" / "processed" / "dvf_final.parquet"

DEPARTEMENTS = [f"{i:02d}" for i in range(1, 96)] + ["2A", "2B"] + [
    "971", "972", "973", "974", "976"
]


# -----------------------------------------------------------------------
# 1. Chargement (mis en cache pour ne pas recharger à chaque interaction)
# -----------------------------------------------------------------------

@st.cache_resource
def charger_modele():
    if not MODELE_PATH.exists():
        return None, None
    modele = joblib.load(MODELE_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return modele, meta


@st.cache_data
def charger_echantillon_dataset(n=20_000):
    if not DATASET_PATH.exists():
        return None
    df = pd.read_parquet(DATASET_PATH)
    if len(df) > n:
        df = df.sample(n, random_state=42)
    return df


# -----------------------------------------------------------------------
# 2. Configuration de la page
# -----------------------------------------------------------------------

st.set_page_config(
    page_title="Prédiction du prix immobilier au m²",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Prédiction du prix immobilier au m² — DVF")
st.caption(
    "Démonstration du modèle entraîné sur les Demandes de Valeurs "
    "Foncières (2021–2025), enrichies avec les données INSEE."
)

modele, meta = charger_modele()

if modele is None:
    st.error(
        f"Aucun modèle trouvé à l'emplacement : `{MODELE_PATH}`\n\n"
        f"Lancez d'abord `06_modelisation_ml.py` (ou `07_modelisation_ml.py`) "
        f"pour entraîner et sauvegarder un modèle."
    )
    st.stop()

onglet_predire, onglet_performance, onglet_donnees = st.tabs(
    ["🔮 Prédire un prix", "📊 Performance du modèle", "🗂️ Explorer les données"]
)


# -----------------------------------------------------------------------
# 3. Onglet Prédiction
# -----------------------------------------------------------------------

with onglet_predire:
    st.subheader("Caractéristiques du bien")

    col1, col2, col3 = st.columns(3)

    with col1:
        type_bien = st.selectbox("Type de bien", ["Appartement", "Maison"])
        surface_habitable = st.number_input(
            "Surface habitable (m²)", min_value=9, max_value=1000, value=65
        )
        nombre_pieces = st.number_input(
            "Nombre de pièces principales", min_value=1, max_value=20, value=3
        )
        surface_terrain = st.number_input(
            "Surface du terrain (m²)", min_value=0, max_value=50_000, value=0
        )

    with col2:
        departement = st.selectbox("Département", DEPARTEMENTS, index=74)  # 75 par défaut
        annee = st.selectbox("Année de la transaction", [2021, 2022, 2023, 2024, 2025], index=4)
        mois = st.slider("Mois", 1, 12, 6)

    with col3:
        latitude = st.number_input("Latitude", value=48.8566, format="%.4f")
        longitude = st.number_input("Longitude", value=2.3522, format="%.4f")
        population = st.number_input(
            "Population de la commune", min_value=0, max_value=3_000_000, value=2_100_000
        )
        revenu_median = st.number_input(
            "Revenu médian de la commune (€/an)", min_value=0, max_value=100_000, value=22_000
        )

    st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=9)

    if st.button("Prédire le prix au m²", type="primary"):
        colonne_population = (
            "population_geo" if "population_geo" in meta["features_num"] else "population_totale"
        )

        entree = pd.DataFrame([{
            "Surface reelle bati": surface_habitable,
            "Nombre pieces principales": nombre_pieces,
            "Surface terrain": surface_terrain,
            "latitude": latitude,
            "longitude": longitude,
            colonne_population: population,
            "revenu_median": revenu_median,
            "annee": annee,
            "mois": mois,
            "Type local": type_bien,
            "Code departement": departement,
        }])

        # On ne garde que les colonnes attendues par le modèle, dans l'ordre
        colonnes_attendues = meta["features_num"] + meta["features_cat"]
        entree = entree[[c for c in colonnes_attendues if c in entree.columns]]

        prediction_log = modele.predict(entree)[0]
        prix_m2_predit = np.expm1(prediction_log)
        prix_total_estime = prix_m2_predit * surface_habitable

        st.success("Estimation réalisée")
        col_a, col_b = st.columns(2)
        col_a.metric("Prix estimé au m²", f"{prix_m2_predit:,.0f} €")
        col_b.metric("Valeur totale estimée du bien", f"{prix_total_estime:,.0f} €")

        st.caption(
            f"Modèle utilisé : **{meta['modele_retenu']}** "
            f"(R² sur l'échelle log : {meta['r2_log']:.3f}). "
            "Cette estimation ne remplace pas une évaluation professionnelle."
        )


# -----------------------------------------------------------------------
# 4. Onglet Performance
# -----------------------------------------------------------------------

with onglet_performance:
    st.subheader("Comparaison des modèles entraînés")

    if COMPARAISON_PATH.exists():
        tableau = pd.read_csv(COMPARAISON_PATH)
        st.dataframe(tableau, use_container_width=True)
    else:
        st.info("Fichier de comparaison introuvable.")

    st.subheader("Graphiques générés lors de l'entraînement")

    graphiques = {
        "Comparaison des modèles": "01_comparaison_modeles.png",
        "Prédictions vs réalité": "02_predictions_vs_reel.png",
        "Analyse des résidus": "03_residus.png",
        "Importance des variables": "04_importance_variables.png",
        "Distribution et évolution des prix": "05_distribution_et_evolution.png",
    }

    for titre, nom_fichier in graphiques.items():
        chemin = GRAPH_DIR / nom_fichier
        if chemin.exists():
            st.markdown(f"**{titre}**")
            st.image(str(chemin), use_container_width=True)
        else:
            st.caption(f"{titre} : image non trouvée ({nom_fichier})")


# -----------------------------------------------------------------------
# 5. Onglet Exploration des données
# -----------------------------------------------------------------------

with onglet_donnees:
    st.subheader("Aperçu du dataset (échantillon)")

    df_echantillon = charger_echantillon_dataset()

    if df_echantillon is None:
        st.info("Dataset introuvable, impossible d'afficher un aperçu.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            departements_dispo = sorted(df_echantillon["Code departement"].unique())
            filtre_dept = st.multiselect(
                "Filtrer par département", departements_dispo, default=[]
            )
        with col2:
            filtre_type = st.multiselect(
                "Filtrer par type de bien",
                sorted(df_echantillon["Type local"].unique()),
                default=[],
            )

        df_filtre = df_echantillon.copy()
        if filtre_dept:
            df_filtre = df_filtre[df_filtre["Code departement"].isin(filtre_dept)]
        if filtre_type:
            df_filtre = df_filtre[df_filtre["Type local"].isin(filtre_type)]

        st.write(f"{len(df_filtre):,} biens affichés")

        col1, col2, col3 = st.columns(3)
        col1.metric("Prix médian au m²", f"{df_filtre['prix_m2'].median():,.0f} €")
        col2.metric("Surface médiane", f"{df_filtre['Surface reelle bati'].median():,.0f} m²")
        col3.metric("Revenu médian moyen", f"{df_filtre['revenu_median'].mean():,.0f} €")

        st.bar_chart(df_filtre.groupby("annee")["prix_m2"].median())

        st.dataframe(
            df_filtre[[
                "Date mutation", "Type local", "Valeur fonciere",
                "Surface reelle bati", "prix_m2", "Code departement",
            ]].head(200),
            use_container_width=True,
        )
