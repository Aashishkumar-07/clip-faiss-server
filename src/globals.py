from dotenv import load_dotenv
import os

load_dotenv()

INDEX_PATH = os.getenv("INDEX_PATH", "./faiss_index")
IMAGE_DIR = os.getenv("IMAGE_DIR", "./images")
THRESHOLD = float(os.getenv("THRESHOLD", 0.95))
CAPTION_IMAGE_SIMILARITY_THRESHOLD = float(os.getenv("CAPTION_IMAGE_SIMILARITY_THRESHOLD", 0.95))
MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-large-patch14")
MODEL_PATH = os.getenv("MODEL_PATH", "~/Desktop/checkpoints/llava-fastvithd_1.5b_stage3")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_API_SECRET_KEY")
BUCKET_NAME  = os.getenv("BUCKET_NAME")
