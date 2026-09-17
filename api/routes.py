import os, shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from src.ingestion import extract_text_with_pages, chunk_documents
from src.vectorstore import store_documents_in_pinecone
from src.rag_engine import ask_research_assistant

router = APIRouter()
UPLOAD_DIR = "data/raw_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        docs = extract_text_with_pages(file_path)
        chunks = chunk_documents(docs)
        # التعديل صار هنا: صار يرفع للسحابة بدال جهازك
        store_documents_in_pinecone(chunks) 
        return {"filename": file.filename, "status": "Indexed successfully", "chunks_count": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        return ask_research_assistant(request.question, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
