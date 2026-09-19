import os
import shutil
import logging
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# استدعاء دوال الذكاء الاصطناعي وقاعدة البيانات
from src.vectorstore import query_pinecone
# from src.vectorstore import store_documents_in_pinecone # أزل علامة # عند تفعيل الرفع لـ Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI

# تحميل متغيرات البيئة
load_dotenv()

# إعداد نظام تسجيل الأخطاء
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Assistant Backend")

# إعداد CORS للواجهة
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# التحقق من مفاتيح البيئة عند التشغيل
@app.on_event("startup")
async def startup_event():
    required_vars = ["PINECONE_API_KEY", "GOOGLE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        error_msg = f"Missing environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info("✓ Environment variables validated")
    logger.info("✓ Application started successfully")

# إنشاء مجلد الملفات المؤقتة
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

# 1. مسار رفع الملفات
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    logger.info(f"📥 Received file upload: {file.filename}")
    
    if not file.filename.lower().endswith('.pdf'):
        logger.warning(f"❌ Invalid file type: {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )
    
    file_size = len(await file.read())
    await file.seek(0)
    
    if file_size > 20 * 1024 * 1024:
        logger.warning(f"❌ File too large: {file.filename} ({file_size} bytes)")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 20MB limit"
        )
    
    file_path = TEMP_DIR / f"temp_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"✓ File saved to disk: {file_path}")
        
        # هنا يتم معالجة الـ PDF ورفعه لـ Pinecone
        # chunks = extract_chunks_from_pdf(str(file_path))
        # store_documents_in_pinecone(chunks)
        
        logger.info(f"✓ Successfully processed {file.filename}")
        return {
            "message": f"Successfully uploaded {file.filename}",
            "filename": file.filename,
            "size": file_size
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing file: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )
    finally:
        await file.close()

# 2. مسار المحادثة والأسئلة (بدون async لمنع التعليق)
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def process_query(request: QueryRequest):
    try:
        # البحث في المستندات
        docs = query_pinecone(request.query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # تجهيز نموذج الذكاء الاصطناعي
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        prompt = f"أجب على السؤال بناءً على المعلومات التالية فقط.\n\nالمعلومات:\n{context}\n\nالسؤال: {request.query}"
        
        # توليد الإجابة
        response = llm.invoke(prompt)
        return {"answer": response.content}
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="حدث خطأ أثناء معالجة السؤال."
        )

# 3. مسار فحص حالة السيرفر
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# تشغيل السيرفر محلياً
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)