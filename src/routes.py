from services.faiss_vectorstore import add_to_index, save_index, search_similar
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from schemas import SearchRequest, ResponseModelSearch
import services.faiss_vectorstore as faiss_vectorstore
from globals import IMAGE_DIR
from pathlib import Path
from PIL import Image
import uuid
import io
import os


# Routes
router = APIRouter()
    
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

@router.get("/")
def root():
  return {"message": "Server is running"}