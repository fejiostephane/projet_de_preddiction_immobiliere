import requests
import os
from tqdm import tqdm

# Dossier de stockage
RAW_DIR = "../data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# URLs DVF (tu peux étendre jusqu’à 2015 ensuite)
DVF_FILES = {
    "2025": "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20260405-002321/valeursfoncieres-2025.txt.zip",
    "2024": "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20260405-002306/valeursfoncieres-2024.txt.zip",
    "2023": "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20260405-002251/valeursfoncieres-2023.txt.zip",
    "2022": "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20260405-002236/valeursfoncieres-2022.txt.zip",
    "2021": "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20260405-002223/valeursfoncieres-2021.txt.zip",
}


def download_file(year, url):
    print(f"\n📥 Téléchargement DVF {year}")

    response = requests.get(url, stream=True)

    file_path = os.path.join(RAW_DIR, f"dvf_{year}.zip")

    total_size = int(response.headers.get("content-length", 0))

    with open(file_path, "wb") as file, tqdm(
        desc=year,
        total=total_size,
        unit="B",
        unit_scale=True
    ) as bar:

        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
            bar.update(len(chunk))

    print(f"✅ {year} terminé → {file_path}")


def main():
    for year, url in DVF_FILES.items():
        download_file(year, url)


if __name__ == "__main__":
    main()