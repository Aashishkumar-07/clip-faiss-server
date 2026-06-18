from services.faiss_vectorstore import add_to_index, save_index, search_similar
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from schemas import SearchRequest, ResponseModelSearch, CaptionResponse
import services.faiss_vectorstore as faiss_vectorstore
import services.fastvlm_caption as fastVLM
from globals import IMAGE_DIR, CAPTION_IMAGE_SIMILARITY_THRESHOLD
from pathlib import Path
from PIL import Image
import numpy as np
import embeddings 
import uuid
import io
import os

# Routes
router = APIRouter()
last_embedding = None

# Helper function
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))

# Deprecated with RabbitMQTT ingestion
@router.post("/create_embedding")
async def process(
    x: float = Form(...),
    y: float = Form(...),
    z: float = Form(...),
    file: UploadFile = File(...)
    ):
  try:
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_id = str(uuid.uuid4())
    filename = f"img_{image_id}.jpg"

    # Save the image
    image_dir = Path(IMAGE_DIR)
    image_dir.mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(IMAGE_DIR, filename)
    image.save(filepath)
    print("saved the imaged")

    coordinate = {"x": x, "y": y, "z": z}

    # Add to FAISS
    success = add_to_index(faiss_vectorstore.vectorstore, coordinate, filename)
    if not success :
        if os.path.exists(filepath):
            os.remove(filepath)
            print("Duplicate image deleted")
        return {
            "message": "Similar embedding already stored",
            "filename": filename,
            "coords": coordinate
        }


    # Save index 
    save_index(faiss_vectorstore.vectorstore)

    return {
        "message": "Embedding stored",
        "filename": filename,
        "coords": coordinate
    }
  
  # Intentionally returned API error (preserves status code)
  except HTTPException:
      raise

  # Unhandled internal error (converts status code to 500)
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

@router.post("/search", response_model=ResponseModelSearch)
async def search(req: SearchRequest):
  try:
    response = search_similar(faiss_vectorstore.vectorstore, req.query, k=req.k)
    print("response from search_similar: ", response)
    for res in response:
        coordinate = res['coords']  
        res['coordinate'] = { "x": coordinate['x'], "y": coordinate['y'], "z": coordinate['z'] }
    return {"Response": response}
  
  # Intentionally returned API error (preserves status code)
  except HTTPException:
    raise

  # Unhandled internal error (converts status code to 500)
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

@router.post("/caption", response_model=CaptionResponse)
async def get_caption(request: Request):
  global last_embedding
  try:
    clip_embedding = embeddings.CLIPEmbeddings()
    image_bytes = await request.body()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # squeeze to remove batch dimension
    embedding =  np.array(clip_embedding.get_image_embedding(image)).squeeze(axis=0)

    if last_embedding is not None:
      similarity = cosine_similarity(embedding, last_embedding)
      if similarity >= CAPTION_IMAGE_SIMILARITY_THRESHOLD:
          return CaptionResponse(caption="", changed=False)
    
    caption = fastVLM.generate_caption(image)
    last_embedding = embedding
    return CaptionResponse(caption=caption, changed=True)
  
  # Intentionally returned API error (preserves status code)
  except HTTPException as e:
    raise

  # Unhandled internal error (converts status code to 500)
  except Exception as e:
    print(e)
    raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
def root():
  return {"message": "Server is running"}