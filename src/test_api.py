import requests

url = "https://data.gouv.fr/api/1/datasets"

response = requests.get(url)

print(response.status_code)