from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered cybersecurity threat detection and monitoring platform.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_url,
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "success": True,
        "message": "DeepShield AI API is running",
        "version": settings.app_version,
    }


@app.get("/api/health")
def health_check():
    return {
        "success": True,
        "data": {
            "service": settings.app_name,
            "status": "healthy",
            "version": settings.app_version,
            "environment": settings.environment,
        },
    }