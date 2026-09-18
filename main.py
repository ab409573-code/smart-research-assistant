import logging
import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging to see errors in Render
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Assistant Backend")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validate environment variables at startup
@app.on_event("startup")
async def startup_event():
    """Validate that all required environment variables are set"""
    required_vars = ["PINECONE_API_KEY", "GOOGLE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        error_msg = f"Missing environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info("✓ Environment variables validated")
    logger.info("✓ Application started successfully")

# Create temp directory if it doesn't exist
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF file and store documents in Pinecone.
    
    Args:
        file: PDF file uploaded from frontend
    
    Returns:
        JSON with success message or error details
    """
    logger.info(f"📥 Received file upload: {file.filename}")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        logger.warning(f"❌ Invalid file type: {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )
    
    # Validate file size (max 20MB)
    file_size = len(await file.read())
    await file.seek(0)  # Reset file pointer
    
    if file_size > 20 * 1024 * 1024:
        logger.warning(f"❌ File too large: {file.filename} ({file_size} bytes)")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 20MB limit"
        )
    
    file_path = TEMP_DIR / f"temp_{file.filename}"
    
    try:
        # Save file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"✓ File saved to disk: {file_path}")
        
        # Process PDF and store in Pinecone
        # Uncomment when vectorstore.py is ready:
        # from vectorstore import store_documents_in_pinecone
        # from pdf_processor import extract_chunks_from_pdf
        # 
        # chunks = extract_chunks_from_pdf(str(file_path))
        # store_documents_in_pinecone(chunks)
        # logger.info(f"✓ Documents stored in Pinecone: {len(chunks)} chunks")
        
        logger.info(f"✓ Successfully processed {file.filename}")
        return {
            "message": f"Successfully uploaded {file.filename}",
            "filename": file.filename,
            "size": file_size
        }
        
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File could not be saved to disk"
        )
    except Exception as e:
        logger.error(f"❌ Error processing file: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )
    finally:
        # Clean up uploaded file object
        await file.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)