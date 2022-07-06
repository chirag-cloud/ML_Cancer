import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'sex':1, 'age':25, 'time':7,'Number_of_Warts':8, 'type':2, 'area':100,'induration_diameter':35})

print(r.json())
