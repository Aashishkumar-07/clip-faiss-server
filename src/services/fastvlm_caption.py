from predict import generate_caption
from globals import IMAGE_DIR
from PIL import Image
import numpy as np
import os

# This func gets called as part of semantic memory creation pipeline (RabbitMQ subsciber image_ingestion pipeline)
def handle_generate_caption(image_path: str):
    filepath = os.path.join(IMAGE_DIR, image_path)
    pil_image = Image.open(filepath).convert("RGB")
    caption = generate_caption(pil_image)
    print("Caption generated from handle_generate_caption:", caption)
    return caption