import os
from dotenv import load_dotenv

load_dotenv()

INDEX_PATH = os.getenv("INDEX_PATH", "./faiss_index")
IMAGE_DIR = os.getenv("IMAGE_DIR", "./images")
THRESHOLD = float(os.getenv("THRESHOLD", 0.90))
MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-large-patch14")