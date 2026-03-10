import requests

url = "http://127.0.0.1:5000/predict"

data = {
 "features":[0.00632,18.0,2.31,0,0.538,6.575,65.2,4.09,1,296,15.3,396.9,4.98]
}

response = requests.post(url, json=data)

print(response.json())