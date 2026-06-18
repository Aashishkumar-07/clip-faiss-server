# Pydantic model
from pydantic import BaseModel, Field

# Request model
class SearchRequest(BaseModel):
  query: str = Field(..., description="user prompt")
  k: int = Field(default=1, description="An optional integer field")

# Response model
class CreateEmbeddingResponse(BaseModel):
  message: str
  filename: str
  coords: list[float]

class SearchResult(BaseModel):
  filename: str
  coordinate: dict
  caption: str
  score: float

class ResponseModelSearch(BaseModel):
    Response: list[SearchResult]

class CaptionResponse(BaseModel):
    caption: str
    changed: bool
