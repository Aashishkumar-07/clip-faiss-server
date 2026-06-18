from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from langchain_core.embeddings import Embeddings
from globals import MODEL_NAME, IMAGE_DIR
from PIL import Image
import torch
import os

# Custom CLIPEmbeddings class implementing the methods in abstract class Embeddings
# smaller model - openai/clip-vit-base-patch32
# Using transformers==4.48.3
class CLIPEmbeddings(Embeddings):
  _instance = None

  # Overriding __new__ to implement singleton pattern
  # Ensures only one instance of CLIPEmbeddings is created
  def __new__(cls, *args, **kwargs): # Responsible for object creation
    if cls._instance is None:
        cls._instance = super().__new__(cls)
    return cls._instance
    
  def __init__(self, device="cpu"):
    if hasattr(self, "_initialized"):  # prevent re-init on second call
        return
  
    print(f"clip embedding model : {MODEL_NAME} loaded on {device}")  
    self.model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    self.tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
    self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    self.device = device
    self._initialized = True

  def get_image_embedding(self, image):
    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
    with torch.no_grad():
      # image_features = self.model.get_image_features(**inputs)
      embedding = self.model.get_image_features(**inputs)

    # print(f"DIM {image_features.pooler_output.shape}")
    # embedding = image_features.pooler_output
    # print(f"embedding shape {embedding.shape}")
    return embedding.cpu().tolist() # converting tensor list to python list

  def get_text_embedding(self, text: str):
    inputs = self.tokenizer([text], padding=True, return_tensors="pt").to(self.device)
    with torch.no_grad():
      # text_features = self.model.get_text_features(**inputs)
      embedding = self.model.get_text_features(**inputs)

    # print(f"DIM {text_features.pooler_output.shape}")
    # print(f"DIM {text_features.shape, text_features}")
    # embedding = text_features.pooler_output
    # print(f"embedding shape {embedding.shape}")
    return embedding.cpu().tolist()  # converting tensor list to python list

  def embed_documents(self, texts: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for text in texts:
      if text.startswith("img_") and text.endswith(".jpg"):
        filepath = os.path.join(IMAGE_DIR, text)
        image = Image.open(filepath).convert("RGB")
        print("calling get_image_embedding")
        embedding = self.get_image_embedding(image)
      else:
        print("calling get_text_embedding")
        embedding = self.get_text_embedding(text)
      embeddings.extend(embedding)

    # print(f"Length of embeddings list: {len(embeddings)}")
    # print(f"Embeddings list contains: {embeddings}")
    return embeddings

  def embed_query(self, text):
    embedding = self.embed_documents([text])[0]
    # print(f"embedding value : {embedding}")
    return embedding