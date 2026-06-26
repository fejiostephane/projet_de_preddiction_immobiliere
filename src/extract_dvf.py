import requests

url = "https://www.data.gouv.fr/api/1/datasets/?q=demandes%20de%20valeurs%20foncieres"

response = requests.get(url)

print(response.status_code)

data = response.json()

print(data["data"][0]["title"])
print(data["data"][0]["id"])