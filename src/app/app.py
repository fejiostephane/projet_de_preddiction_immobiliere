import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --- Config de la page (doit être la 1re commande Streamlit) ---
st.set_page_config(page_title="Estimateur immobilier", page_icon="🏠", layout="centered")

# --- Localisation des artefacts (chemin robuste, relatif au script) ---
BASE = Path(__file__).resolve().parent            # src/app
ML = BASE.parent.parent / "data" / "processed" / "ml"

@st.cache_resource
def charger_artefacts():
    modele         = joblib.load(ML / "modele_final_lightgbm.pkl")
    features_order = joblib.load(ML / "features_order.pkl")
    prix_dept      = joblib.load(ML / "prix_dept.pkl")
    median_terrain = joblib.load(ML / "median_terrain.pkl")
    global_log     = joblib.load(ML / "global_log.pkl")
    communes       = pd.read_parquet(ML / "communes_ref.parquet")
    return modele, features_order, prix_dept, median_terrain, global_log, communes

modele, features_order, prix_dept, median_terrain, global_log, communes = charger_artefacts()

# --- En-tête ---
st.title("🏠 Estimateur de prix immobilier")
st.caption("Modèle LightGBM entraîné sur 5,6M transactions DVF (2021-2025)")

# =========================================================
#  BARRE LATÉRALE — saisie des caractéristiques du bien
# =========================================================
st.sidebar.header("📋 Caractéristiques du bien")

# --- Localisation : département puis commune (liste filtrée) ---
depts = sorted(communes["code_departement"].unique())
dept = st.sidebar.selectbox("Département", depts, index=depts.index("75") if "75" in depts else 0)

communes_dept = communes[communes["code_departement"] == dept].sort_values("nom").reset_index(drop=True)
choix = st.sidebar.selectbox(
    "Commune",
    communes_dept.index,
    format_func=lambda i: communes_dept.loc[i, "nom"]      # affiche le nom, garde la ligne
)
commune = communes_dept.loc[choix]                          # la ligne complète de la commune

# --- Caractéristiques du bien ---
type_local      = st.sidebar.radio("Type de bien", ["Appartement", "Maison"])
surface_bati    = st.sidebar.number_input("Surface habitable (m²)", 9, 1000, 70, step=5)
nb_pieces       = st.sidebar.number_input("Nombre de pièces", 1, 20, 3, step=1)
surface_terrain = st.sidebar.number_input("Surface terrain (m²)", 0, 10000, 0, step=50,
                                          help="0 pour un appartement")

col1, col2 = st.sidebar.columns(2)
annee = col1.selectbox("Année", [2021, 2022, 2023, 2024, 2025], index=4)
mois  = col2.selectbox("Mois", list(range(1, 13)), index=5)

# --- Bouton d'estimation ---
estimer = st.sidebar.button("💶 Estimer le prix", type="primary", use_container_width=True)

# =========================================================
#  PRÉDICTION
# =========================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

if not estimer:
    st.info("👈 Renseigne les caractéristiques dans la barre latérale, puis clique sur **Estimer le prix**.")
else:
    # Maison sans terrain saisi -> médiane (comme à l'entraînement)
    terrain = surface_terrain
    if type_local == "Maison" and terrain == 0:
        terrain = median_terrain

    # Reconstruction des 20 features, à l'identique de l'entraînement
    feat = {
        "Surface reelle bati": surface_bati,
        "Nombre pieces principales": nb_pieces,
        "Surface terrain": terrain,
        "annee": annee,
        "mois": mois,
        "codeRegion": commune["codeRegion"],
        "population_geo": commune["population_geo"],
        "longitude": commune["longitude"],
        "latitude": commune["latitude"],
        "revenu_median": commune["revenu_median"],
        "surface_totale": surface_bati + terrain,
        "surface_par_piece": surface_bati / nb_pieces,
        "date_num": annee + (mois - 1) / 12,
        "surface_x_revenu": surface_bati * commune["revenu_median"],
        "distance_paris": haversine(commune["latitude"], commune["longitude"], 48.8566, 2.3522),
        "is_paris": int(dept == "75"),
        "is_petite_couronne": int(dept in ["75", "92", "93", "94"]),
        "prix_median_dept": prix_dept.get(dept, prix_dept.median()),
        "prix_median_commune": commune["prix_median_commune"],
        "is_maison": int(type_local == "Maison"),
    }

    # DataFrame d'1 ligne, colonnes dans le BON ordre
    X_new = pd.DataFrame([feat])[features_order]

    # Prédiction (log) -> euros
    prix = float(np.expm1(modele.predict(X_new)[0]))
    prix_m2 = prix / surface_bati

    # --- Affichage du résultat ---
    st.subheader(f"📍 {commune['nom']} ({dept}) — {type_local} de {surface_bati} m²")

    c1, c2 = st.columns(2)
    c1.metric("💶 Prix estimé", f"{prix:,.0f} €".replace(",", " "))
    c2.metric("📐 Prix au m²", f"{prix_m2:,.0f} €/m²".replace(",", " "))

    st.caption("⚠️ Fourchette indicative : ± ~40 000 € (erreur médiane du modèle). "
               "Estimation statistique, ne remplace pas une expertise.")
    st.balloons()
