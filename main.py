from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router # أو حسب مسار الاستدعاء لديك

app = FastAPI()

# هذا الكود يسمح للسيرفر باستقبال الطلبات من أي واجهة خارجية
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)