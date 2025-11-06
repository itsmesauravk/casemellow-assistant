import os
import json
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def get_embedding(text:str):
    """
    Generates and embedding for a given text using Gemini API.
    """

    if not text or text.strip() == "":
        return None
    
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

   

    return result.embeddings[0].values


def save_json(data, file_path):
    """
    Saves data to a JSON file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(file_path):
    """
    Loads data from a JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    


