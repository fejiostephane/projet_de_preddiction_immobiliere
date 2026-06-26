import requests

dataset_id = "5c4ae55a634f4117716d5656"

url = f"https://www.data.gouv.fr/api/1/datasets/{dataset_id}/"

response = requests.get(url)

data = response.json()

for resource in data["resources"]:
    print("\n---")
    print(resource["title"])
    print(resource["format"])
    print(resource["url"])