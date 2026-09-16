from fastapi import FastAPI

app = FastAPI(
    title="Smart Research Assistant API",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {
        "status": "active",
        "project": "Smart Research Assistant",
        "message": "Backend server is running smoothly!"
    }