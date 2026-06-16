import services.faiss_vectorstore as faiss_vectorstore
import services.supabase_client as supabase_client
from globals import IMAGE_DIR, BUCKET_NAME
from pathlib import Path
from PIL import Image
import os
import io

def download_image(image_path: str):
    image_bytes = supabase_client.supabase_client.storage.from_(BUCKET_NAME).download(image_path)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    print("Downloaded the image from Supabase storage")

    # Save the image locally
    image_dir = Path(IMAGE_DIR)
    image_dir.mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(IMAGE_DIR,  image_path)
    image.save(filepath)
    print(f"saved the image locally for processing: {filepath}")
    return filepath

def delete_image(image_path: str):
    response = supabase_client.supabase_client.storage.from_(BUCKET_NAME).remove([image_path])
    print(f"Deleted the image from Supabase storage: {image_path}, response: {response}")

def create_embedding(image_path: str, coordinate: dict):
    filepath = download_image(image_path)
    success = faiss_vectorstore.add_to_index(faiss_vectorstore.vectorstore, coordinate, image_path)

    if os.path.exists(filepath):
        os.remove(filepath)
        print("Locally saved image deleted")

    if not success :
        delete_image(image_path)
    else:
        # Save index 
        faiss_vectorstore.save_index(faiss_vectorstore.vectorstore)
    
