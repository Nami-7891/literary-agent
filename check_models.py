import os
import requests
from dotenv import load_dotenv

# 1. Load the API key from your .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: Could not find GOOGLE_API_KEY in your .env file.")
    exit()

# 2. Ask Google's servers for the official list of models
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
response = requests.get(url)

# 3. Print the results cleanly
if response.status_code == 200:
    data = response.json()
    print("\n--- Available Models for Text Generation ---")
    for model in data.get("models", []):
        # We only care about models that support 'generateContent' (text generation)
        if "generateContent" in model.get("supportedGenerationMethods", []):
            # Strip the 'models/' prefix so you know exactly what to paste into LangChain
            exact_name = model['name'].replace("models/", "")
            print(f"- {exact_name}")
else:
    print(f"Error connecting to Google API: {response.status_code}\n{response.text}")