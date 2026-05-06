from faiss_vectorstore import add_to_index, save_index, search_similar, load_or_create_index
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from schemas import SearchRequest, ResponseModelSearch
from contextlib import asynccontextmanager
from globals import IMAGE_DIR
from pathlib import Path
from PIL import Image
import uuid
import io
import os

# Global variables
vectorstore = None

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore
    vectorstore = load_or_create_index() 
    print("Model + FAISS ready!")
    yield

# Initialization
app = FastAPI(lifespan=lifespan)

# Routes
@app.post("/create_embedding")
async def process(
    x: float = Form(...),
    y: float = Form(...),
    z: float = Form(...),
    caption: str = Form(...),
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

    coordinate = [x, y, z]

    # Add to FAISS
    success = add_to_index(vectorstore, coordinate, caption, filename)
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
    save_index(vectorstore)

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


@app.post("/search", response_model=ResponseModelSearch)
async def search(req: SearchRequest):
  try:
    response = search_similar(vectorstore, req.query, k=req.k)
    for res in response:
        coordinate = res.pop('coords')  
        res['coordinate'] = { "x": coordinate[0], "y": coordinate[1], "z": coordinate[2] }
    print("response from search_similar: ", response)
    return {"Response": response}
  
  # Intentionally returned API error (preserves status code)
  except HTTPException:
    raise

  # Unhandled internal error (converts status code to 500)
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
  return {"message": "Server is running"}
