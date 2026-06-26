import requests

url = "https://www.data.gouv.fr/api/1/datasets/?q=niveau+de+vie+commune"

data = requests.get(url).json()

for d in data["data"]:
    print("\n---")
    print(d["title"])
    print(d["id"])