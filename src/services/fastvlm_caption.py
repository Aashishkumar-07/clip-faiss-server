from predict import generate_caption
from globals import IMAGE_DIR
from PIL import Image
import numpy as np
import os

def handle_generate_caption(image_path: str):
    filepath = os.path.join(IMAGE_DIR, image_path)
    image = Image.open(filepath).convert("RGB")
    cv_image = np.array(image)
    caption = generate_caption(cv_image=cv_image)
    print("Caption generated from handle_generate_caption:", caption)
    return caption