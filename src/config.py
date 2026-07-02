from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv() 

    
api_key = os.getenv("OPENAI_API_KEY")
hf_key = os.getenv("HUGGING_FACE_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Add it to .env and re-run.")

client = OpenAI(base_url='https://api.gapgpt.app/v1', api_key=api_key, timeout=300.0, 
    max_retries=2)
MODEL_NAME = "gpt-4o-mini"

Embed_model="text-embedding-3-small"

proxies = {
    "http": "http://127.0.0.1:10886",
    "https": "http://127.0.0.1:10886"
}
