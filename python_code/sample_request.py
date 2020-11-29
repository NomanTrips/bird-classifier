import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('../src/assets/hawk-three.jpg','rb')})
print(resp.json())