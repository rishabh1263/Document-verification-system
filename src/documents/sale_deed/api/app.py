"""
FastAPI Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

# ==========================================================
# FastAPI Application
# ==========================================================

app = FastAPI(
    title="Sale Deed Verification API",
    description="""
AI-powered API for verifying and extracting information from Indian Sale Deeds.

Features:
- Document Upload
- OCR Processing
- Document Verification
- AI-based Field Extraction
- Document Classification
""",
    version="1.0.0",
    contact={
        "name": "AI Team"
    }
)

# ==========================================================
# Enable CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# Register API Routes
# ==========================================================

app.include_router(
    router,
    prefix="/api/v1",
    tags=["Sale Deed Verification"]
)

# ==========================================================
# Health Check
# ==========================================================

@app.get("/", tags=["Health"])
def home():
    return {
        "status": "success",
        "message": "Sale Deed Verification API is running."
    }

# ==========================================================
# Startup Event
# ==========================================================

@app.on_event("startup")
def startup_event():
    print("=" * 60)
    print("Sale Deed Verification API Started")
    print("Swagger UI: http://127.0.0.1:8000/docs")
    print("=" * 60)
