import requests
import os
from dotenv import load_dotenv

load_dotenv()


API_KEY = os.getenv("GEMINI_API_KEY")
print("key loaded:" , bool(API_KEY))


url = "https://generativelanguage.googleapis.com/v1beta/models"

headers = {
    "x-goog-api-key" : API_KEY
}

response = requests.get(url,headers=headers)



if response.status_code == 200:
    data = response.json()

    for model in data.get("models" , []):
        print("model:",model["name"])
        print("display:",model.get("displayName"))

else:
    print(response.status_code)
    print(response.text)