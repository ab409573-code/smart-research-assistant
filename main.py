from fastapi import FastAPI
from api.routes import router  # استدعاء مسارات الذكاء الاصطناعي

app = FastAPI(
    title="Smart Research Assistant API",
    version="1.0.0"
)

# دمج مسارات الذكاء الاصطناعي مع التطبيق الرئيسي تحت رابط /api
app.include_router(router, prefix="/api")

@app.get("/health")
def health_check():
    return {
        "status": "active",
        "project": "Smart Research Assistant",
        "message": "Backend server is running smoothly!"
    }