# 🏠 Prédiction des Prix Immobiliers en France (DVF)

Projet de Data Science de bout en bout : **collecte → ETL → base de données → analyse → machine learning → dashboard**, avec pour objectif de **prédire le prix d'un bien immobilier en France** à partir des données publiques DVF (Demandes de Valeurs Foncières).

> **Stack technique :** Python · pandas · scikit-learn · XGBoost · LightGBM · SHAP · MySQL · SQLAlchemy · Streamlit

---

## 📌 Table des matières

1. [Objectif du projet](#-objectif-du-projet)
2. [Architecture générale](#-architecture-générale)
3. [Sources de données](#-sources-de-données)
4. [Télécharger le jeu de données (collecte)](#-télécharger-le-jeu-de-données-collecte)
5. [Structure du dépôt](#-structure-du-dépôt)
6. [Installation & prérequis](#-installation--prérequis)
7. [Le pipeline étape par étape](#-le-pipeline-étape-par-étape)
8. [Résultats](#-résultats)
9. [Choix méthodologiques clés](#-choix-méthodologiques-clés)
10. [Limites & pistes d'amélioration](#-limites--pistes-daméliration)
11. [Comment lancer le projet](#-comment-lancer-le-projet)
12. [Auteur](#-auteur)

---

## 🎯 Objectif du projet

Construire une chaîne complète et rigoureuse, de la donnée brute jusqu'à une application utilisable :

```
Collecte  →  ETL  →  Data Warehouse (MySQL)  →  EDA  →  Feature Engineering
        →  Machine Learning  →  Évaluation  →  Modèle final  →  Dashboard
```

**Objectif final :** estimer la valeur foncière (prix de vente) d'un logement (maison ou appartement) en France métropolitaine et DOM, sur la période **2021 – 2025**.

---

## 🏗️ Architecture générale

```
   Données brutes (DVF, INSEE, geo.api.gouv.fr)
                    │
                    ▼
        ┌───────────────────────┐
        │   ETL (src/etl/)      │  extraction ZIP → nettoyage → consolidation
        │                       │  → enrichissement géo & socio-économique
        └───────────┬───────────┘
                    ▼
        data/processed/dvf_final.parquet   (5,6 M lignes, 23 colonnes)
                    │
                    ▼
        ┌───────────────────────┐
        │   MySQL                │  base `immobilier_dvf`, table `transactions`
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────────────────────────────┐
        │   Data Science (src/ml/)                       │
        │   01 EDA → 02 Préprocessing → 03 Feature Eng.  │
        │   → 04 Modélisation → 05 Évaluation            │
        │   → 06 Modèle final (5,6 M) → 07 Artefacts     │
        └───────────┬───────────────────────────────────┘
                    ▼
        ┌───────────────────────┐
        │  Dashboard Streamlit   │  src/app/app.py
        └───────────────────────┘
```

---

## 🗃️ Sources de données

| Source | Contenu | Format | Emplacement |
|---|---|---|---|
| **DVF** (data.gouv.fr) | Transactions immobilières 2021-2025 | ZIP → TXT (séparateur `\|`) | `data/raw/dvf/` |
| **Référentiel communes** (geo.api.gouv.fr) | Nom, code, département, région, population, coordonnées GPS | JSON | `data/raw/geo/communes_france.json` |
| **Population communes** (INSEE) | Population par commune | CSV | `data/raw/insee/population_communes.csv` |
| **Revenus / niveau de vie** (INSEE - Figaro) | Revenu médian par commune | CSV | `data/raw/insee/revenus_communes.csv` |

---

## 📥 Télécharger le jeu de données (collecte)

Toutes les données du projet sont **publiques et gratuites**. Elles se récupèrent avec les scripts Python du dossier `src/`. Il n'y a **rien à télécharger à la main** : chaque script appelle directement l'API ou l'URL de la source et écrit le fichier dans `data/raw/`.

> ⚠️ **Important — d'où lancer les scripts ?**
> Les scripts utilisent des chemins **relatifs** (`../data/raw/...`). Il faut donc les exécuter **depuis le dossier `src/`**, sinon les fichiers ne se rangeront pas au bon endroit.
> Prérequis : `pip install -r requirements.txt` (les scripts utilisent `requests` et `tqdm`).

### 1. Scripts de téléchargement (ceux qui produisent les données)

| Script (`src/`) | Ce qu'il télécharge | Source | Fichier(s) produit(s) |
|---|---|---|---|
| **`download_dvf.py`** | 🎯 Le cœur du projet : les **5 années de transactions DVF** (2021 → 2025) | data.gouv.fr | `data/raw/dvf_2021.zip` … `dvf_2025.zip` |
| **`referenciel_des_comunes.py`** | Référentiel géographique des communes (nom, code, département, région, population, coordonnées GPS) | geo.api.gouv.fr | `data/raw/geo/communes_france.json` |
| **`revenu_comunes_data.py`** | Revenu médian / niveau de vie par commune | INSEE (Figaro) | `data/raw/insee/revenus_communes.csv` |
| **`test_insee_population.py`** | Population légale par commune | opendata | `data/raw/insee/population_communes.csv` |

### 2. Scripts d'exploration (facultatifs — ne téléchargent aucune donnée)

Ceux-là ont servi à **découvrir les URLs** des jeux de données sur l'API data.gouv.fr. Ils affichent seulement des informations dans la console ; tu n'as **pas besoin de les relancer** pour reconstituer le dataset.

| Script (`src/`) | Rôle |
|---|---|
| `test_api.py` | Teste la connexion à l'API data.gouv.fr (code de statut HTTP) |
| `extract_dvf.py` | Recherche le jeu de données DVF sur l'API et affiche son titre + son `id` |
| `list_dvf_ressources.py` | Liste toutes les ressources (URLs des fichiers) du jeu DVF |
| `revenu_communes.py` | Recherche les jeux de données « niveau de vie / commune » sur l'API |

### 3. Tout télécharger en une fois

```bash
# Depuis la racine du projet
cd src

# 1. DVF — les 5 années de transactions (fichiers lourds, ~1 Go au total, barre de progression)
python download_dvf.py

# 2. Référentiel géographique des communes
python referenciel_des_comunes.py

# 3. Revenu médian par commune (INSEE)
python revenu_comunes_data.py

# 4. Population par commune
python test_insee_population.py

cd ..
```

Une fois ces 4 scripts terminés, le dossier `data/raw/` contient toutes les données brutes nécessaires. On peut alors enchaîner sur la **Phase 2 — ETL** (`src/etl/`) pour décompresser, nettoyer et consolider.

---

## 📂 Structure du dépôt

```
projet_final/
├── data/
│   ├── raw/                       # données brutes téléchargées
│   │   ├── dvf/                   # dvf_2021.zip … dvf_2025.zip
│   │   ├── geo/                   # communes_france.json
│   │   └── insee/                # population & revenus
│   ├── extracted/dvf/            # fichiers DVF décompressés (.txt)
│   └── processed/
│       ├── dvf_final.parquet     # dataset final consolidé (5,6 M lignes)
│       ├── dvf_final_mysql.csv   # export pour import MySQL
│       └── ml/                   # artefacts ML (modèles, encodeurs, jeux)
├── src/
│   ├── etl/                       # pipeline ETL
│   │   ├── 01_extract_dvf_zip.py
│   │   ├── 02_inspect_dvf.py
│   │   ├── 03_nettoyage_dvf.ipynb
│   │   ├── 04_consolidation_dvf.ipynb
│   │   └── transfert_bd.ipynb     # export Parquet → MySQL
│   ├── ml/                        # partie Data Science
│   │   ├── 01_eda.ipynb
│   │   ├── 02_preprocessing.ipynb
│   │   ├── 03_feature_engineering.ipynb
│   │   ├── 04_modelisation.ipynb
│   │   ├── 05_evaluation.ipynb
│   │   ├── 06_modele_final.ipynb
│   │   └── 07_export_artifacts.ipynb
│   ├── app/
│   │   └── app.py                 # dashboard Streamlit
│   └── *.py                       # scripts de collecte (DVF, INSEE, geo)
├── requirements.txt
├── .env                           # identifiants MySQL (NON versionné)
└── readme.md
```

---

## ⚙️ Installation & prérequis

**Prérequis :** Python 3.11+, un serveur MySQL local.

```bash
# 1. Environnement virtuel
python -m venv myvenv
myvenv\Scripts\activate        # Windows

# 2. Dépendances
pip install -r requirements.txt

# 3. Fichier .env à la racine (identifiants MySQL)
```

Contenu du fichier `.env` (à créer, **non versionné**) :

```
DB_USER=root
DB_PASSWORD=********
DB_HOST=localhost
DB_PORT=3306
DB_NAME=immobilier_dvf
```

---

## 🔄 Le pipeline étape par étape

### Phase 1 — Collecte des données
Téléchargement des 5 années de DVF (2021-2025), du référentiel des communes (geo.api.gouv.fr), de la population et des revenus médians (INSEE). Scripts dans `src/`.
👉 Détail des scripts et commandes : voir [Télécharger le jeu de données (collecte)](#-télécharger-le-jeu-de-données-collecte).

### Phase 2 — ETL (`src/etl/`)
- **Extraction** : décompression des ZIP DVF → fichiers `.txt`.
- **Nettoyage** :
  - Filtre `Nature mutation = "Vente"` et `Type local ∈ {Maison, Appartement}`.
  - Conversion des types (virgule décimale → point, dates en `datetime`).
  - Suppression des prix ≤ 1 000 €, surfaces ≤ 5 m², doublons, valeurs extrêmes.
- **Consolidation** : fusion des 5 années.
- **Enrichissement** :
  - Géographique (latitude, longitude, population, région) via le référentiel communes.
  - Socio-économique (revenu médian) via l'INSEE.
- **Sortie** : `data/processed/dvf_final.parquet` — **5 624 088 lignes × 23 colonnes**.

### Phase 3 — Base de données MySQL
Import du dataset dans MySQL (base `immobilier_dvf`, table `transactions`) via `transfert_bd.ipynb`. La partie ML **repart de MySQL**, plus du Parquet.

**Comment on charge les 5,6 M lignes (`transfert_bd.ipynb`) — 4 étapes :**

| Étape | Action | Pourquoi |
|---|---|---|
| 1 | `pd.read_parquet(...)` | on charge le dataset final propre |
| 2 | `df.to_csv(..., sep=";")` | ⚠️ MySQL **ne lit pas le Parquet** ; son chargeur rapide lit du **texte/CSV** |
| 3 | `df.head(0).to_sql(...)` | crée la **structure vide** de la table (pandas déduit le schéma) |
| 4 | `LOAD DATA LOCAL INFILE` | insère les 5,6 M lignes **en une seule opération** (quelques secondes) |

> **Pourquoi `LOAD DATA` et pas `df.to_sql()` pour tout ?** `to_sql` insère par petits paquets via des milliers de requêtes `INSERT` → beaucoup trop lent sur 5,6 M lignes. `LOAD DATA LOCAL INFILE` est le chargeur natif de MySQL : il lit le CSV côté serveur en masse. On utilise donc `to_sql` **uniquement pour créer la table vide** (étape 3), et `LOAD DATA` **pour les données** (étape 4).

**Pourquoi passer par MySQL plutôt que lire directement le Parquet ?**

| Parquet (fichier) | MySQL (Data Warehouse) |
|---|---|
| Fichier sur disque, mono-utilisateur | **Service** interrogeable, multi-utilisateur |
| On charge **tout** en mémoire | On fait du **SQL** : `WHERE`, `LIMIT`, `RAND()`, `GROUP BY`… côté serveur |
| Pas de requêtes | Filtrage / échantillonnage **sans tout charger** |

Concrètement, l'EDA échantillonne directement en base (`SELECT * FROM transactions WHERE RAND() < 0.04 LIMIT 200000`) sans charger les 5,6 M lignes. Le Parquet est le **résultat de l'ETL** ; MySQL est le **Data Warehouse** qui expose ces données de façon interrogeable — la partie ML repart de la base pour simuler un environnement de production.

**Schéma de la table `transactions` (23 colonnes) :**

| Colonne | Description |
|---|---|
| `Valeur fonciere` | 🎯 **cible** — prix de la transaction (€) |
| `Type local` | Maison / Appartement |
| `Surface reelle bati` | surface habitable (m²) |
| `Nombre pieces principales` | nombre de pièces |
| `Surface terrain` | surface du terrain (m², nul pour les appartements) |
| `prix_m2` | prix au m² (dérivé — écarté du ML, fuite de données) |
| `Code departement`, `Code postal`, `Commune`, `codeRegion` | localisation |
| `code_commune_geo`, `nom_commune_geo` | code / nom INSEE de la commune |
| `population_geo`, `revenu_median` | contexte socio-économique |
| `longitude`, `latitude` | coordonnées GPS |
| `annee`, `mois`, `trimestre` | dérivées temporelles |
| `Date mutation`, `Nature mutation` | date et nature (Vente) |

### Notebook 01 — EDA (`src/ml/01_eda.ipynb`)
Analyse exploratoire depuis MySQL (échantillon aléatoire de 200 k lignes) :
- Dimensions, types, valeurs manquantes, statistiques descriptives.
- Distributions, **choix de la cible** : `Valeur fonciere` transformée en `log` (asymétrie ramenée de 28 à 0,31).
- Corrélations, boxplots, analyses métier.

**Réponses métier obtenues :**
- Départements les plus chers : **75 (Paris) > 92 > 94**.
- Les **appartements** sont plus chers au m² que les maisons.
- Le **revenu médian** de la commune influence le prix (corrélation 0,36 sur la cible log).
- Prix en **baisse 2022 → 2024** (hausse des taux d'intérêt) ; 2025 partiellement renseigné.

### Notebook 02 — Préprocessing (`src/ml/02_preprocessing.ipynb`)
- Échantillon de travail reproductible (~563 k lignes, `RAND(42)`).
- Suppression de la **fuite de données** (`prix_m2`), des colonnes constantes et redondantes → **23 → 12 features**.
- Imputation de `Surface terrain` : appartements → 0 ; maisons manquantes → **médiane du train** (anti-fuite).
- **Train/test split 80/20 réalisé AVANT toute transformation apprise.**

### Notebook 03 — Feature Engineering (`src/ml/03_feature_engineering.ipynb`)
Création de **8 nouvelles variables** (12 → 20 features) :

| Feature | Idée |
|---|---|
| `surface_totale` | bâti + terrain |
| `surface_par_piece` | taille moyenne d'une pièce |
| `date_num` | temps continu (tendance des prix) |
| `surface_x_revenu` | interaction surface × revenu |
| `distance_paris` | distance GPS au centre de Paris (Haversine) |
| `is_paris`, `is_petite_couronne` | zones premium |
| `prix_median_commune` | **encodage par la cible, en out-of-fold (K-fold)** sur le log du prix |

### Notebook 04 — Modélisation (`src/ml/04_modelisation.ipynb`)
Comparaison progressive de 6 modèles (métrique principale : **R² sur l'échelle log**, stable ; MedAE en euros) :

| Modèle | R² (log) |
|---|---|
| Régression Linéaire (baseline) | 0,34 |
| Arbre de décision | 0,53 |
| Random Forest | 0,59 |
| Gradient Boosting | 0,59 |
| LightGBM | 0,60 |
| **XGBoost** | **0,61** |

### Notebook 05 — Évaluation (`src/ml/05_evaluation.ipynb`)
- Graphe Prédit vs Réel (régression vers la moyenne aux extrêmes).
- **Importance des variables** + **SHAP** : le prix est piloté par la valeur de localisation (`prix_median_commune`), la surface et l'interaction surface × revenu.
- Analyse des erreurs par tranche de prix (courbe en U) et par type de bien (~22 % d'erreur médiane).

### Notebook 06 — Modèle final (`src/ml/06_modele_final.ipynb`)
Ré-entraînement sur **l'intégralité des 5,6 M lignes** :
- Optimisation mémoire (float32 + category) : **1 542 → 245 Mo (-84 %)**.
- Modèle final **LightGBM** (rapide sur gros volume, sans normalisation).

### Notebook 07 — Export des artefacts (`src/ml/07_export_artifacts.ipynb`)
Export des éléments nécessaires à l'inférence du dashboard : modèle, ordre des features, table de référence des communes, encodeurs (médiane terrain, prix par département, encodage commune).

### Dashboard Streamlit (`src/app/app.py`)
Interface interactive : l'utilisateur choisit une **commune** et saisit les caractéristiques du bien (type, surface, pièces, terrain, date) ; l'application reconstruit les 20 features et affiche le **prix estimé** et le **prix au m²**.


## 📊 Résultats

| Modèle | Données | R² (log) | MedAE | Erreur médiane |
|---|---|---|---|---|
| XGBoost (dev) | 563 k lignes | 0,607 | ~42 k€ | ~22 % |
| **LightGBM (final)** | **5,6 M lignes** | **0,625** | **~41 k€** | **~21 %** |

**Exemple de prédiction (dashboard) :** appartement 70 m², 3 pièces, Paris (75), 2025 → **≈ 662 000 € (9 450 €/m²)** — cohérent avec le marché réel parisien.

---

## 🧠 Choix méthodologiques clés

1. **Cible en log** : les prix immobiliers sont très asymétriques ; on modélise `log1p(prix)` et on évalue le R² sur cette échelle (stable), avec la MAE/MedAE en euros pour l'interprétation.
2. **Anti-fuite de données** : split *avant* toute transformation ; imputation, normalisation et encodages appris **sur le train uniquement**.
3. **Suppression de `prix_m2`** : dérivée de la cible → fuite évidente.
4. **Encodage par la cible en out-of-fold** : diagnostiqué comme surapprentissage via l'importance des variables, puis corrigé par un encodage K-fold (chaque ligne encodée sans utiliser son propre prix).
5. **Développement sur échantillon, entraînement final sur tout** : itération rapide en dev, performance maximale en production.

---

## ⚠️ Limites & pistes d'amélioration

**Limites :**
- **Plafond de performance ~0,62** : variance irréductible du prix *total* (ventes multi-lots, état du bien absent de DVF).
- **Régression vers la moyenne** : le modèle sous-estime le luxe et surestime le bas de gamme.
- `revenu_median` provient de données INSEE relativement anciennes.
- Pas de prédiction fiable au-delà de 2025 (les modèles à base d'arbres n'extrapolent pas).

**Pistes :**
- Détecter et écarter les ventes multi-lots pour réduire le bruit.
- Ajouter des données externes (DPE, proximité transports, écoles).
- Optimisation fine des hyperparamètres (GridSearch / Optuna).
- Déploiement du dashboard (Streamlit Cloud) + persistance complète des encodeurs.

---

## ▶️ Comment lancer le projet

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. (ETL) exécuter les notebooks src/etl/ dans l'ordre puis transfert_bd.ipynb
#    -> alimente la base MySQL immobilier_dvf

# 3. (ML) exécuter les notebooks src/ml/ 01 → 07 dans l'ordre

# 4. Lancer le dashboard (depuis la racine du projet)
streamlit run src/app/app.py
```

---

