import requests

#resp = requests.post("http://localhost:5000/predict",
#                     files={"file": open('../src/assets/hawk-three.jpg','rb')})
resp = requests.post("http://52.37.86.143:5000/predict",
                     files={"file": open('../src/assets/avatars/eagle-avatar.png','rb')})
print(resp.json())