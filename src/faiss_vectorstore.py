from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from globals import THRESHOLD, INDEX_PATH 
from embeddings import CLIPEmbeddings
import pickle
import faiss
import uuid
import torch
import os

# FAISS core logic
def load_or_create_index():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  clip_embeddings = CLIPEmbeddings(device=device)

# If index existed - load
  if os.path.exists(INDEX_PATH):
    index_file = os.path.join(INDEX_PATH, "my_faiss_index.faiss")
    store_file = os.path.join(INDEX_PATH, "my_faiss_index.pkl")
    if os.path.exists(index_file) and os.path.exists(store_file):
      print("Loading existing FAISS index...")
      # return FAISS.load_local(folder_path=INDEX_PATH, embeddings=clip_embeddings, index_name = "my_faiss_index", allow_dangerous_deserialization=True)

      index = faiss.read_index(index_file)
      with open(store_file, 'rb') as file:
        docstore, index_to_docstore_id = pickle.load(file)
      return FAISS(embedding_function = clip_embeddings, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, normalize_L2=True)
    
  # If index does not exist - create
  # distance_strategy = DistanceStrategy.DOT_PRODUCT
  print("Creating new FAISS index...")
  index = faiss.IndexFlatIP(len(clip_embeddings.get_text_embedding("test")[0]))
  return FAISS(
        embedding_function=clip_embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
        normalize_L2 = True,  # cosine similarity via normalized vectors
    )

def get_max_similarity_score(vectorstore: FAISS, img_filename: str, k: int):
    results = vectorstore.similarity_search_with_score(query=img_filename, k=k)
    print(f"Results from vectorstore.similarity_search_with_score function : {results}")

    if not results:
        return None

    scores = [round(score, 3) for _, score in results]
    print(f"Similarity scores: {scores}")

    _, max_score = results[0]
    return max_score 

def add_to_index(vectorstore: FAISS, coords: list, caption: str, img_filename: str):
  max_score = get_max_similarity_score(vectorstore, img_filename, k=3)

  if max_score is not None and max_score > THRESHOLD:
    print(f"Similar embedding (score={max_score:.3f}) already exists")
    return False

  dissimilarity_score = 1 - max_score if max_score is not None else 0
  docs = [
    Document(
        page_content=img_filename,
        metadata=
        {
            "filename": img_filename,
            "coords": coords,
            "caption": caption,
            "dissimilarity_score": dissimilarity_score
        }
    )
  ]

  id = str(uuid.UUID(os.path.splitext(img_filename)[0].split("_", 1)[1]))
  vectorstore.add_documents(documents=docs, ids=[id])
  return True

def save_index(vectorstore):
  vectorstore.save_local(folder_path=INDEX_PATH, index_name="my_faiss_index")
  print("Index saved!")

def search_similar(vectorstore: FAISS, user_query: str, k: int):
  results = vectorstore.similarity_search_with_score(query=user_query, k=k)
  print(f"similarity scores from vectorstore.similarity_search_with_score : {[round(score, 3) for _, score in results]}")

  _, max_matching_score = results[0]
  print(f"Max Matching score : {max_matching_score}")

  return [
      {
          "filename": doc.metadata["filename"],
          "coords": doc.metadata["coords"],
          "caption": doc.metadata["caption"],
          "score": score,

      }
      for doc, score in results
  ]