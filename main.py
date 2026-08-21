from __future__ import annotations

from fastapi import FastAPI

from src.documents.itr.api.routes import router as itr_router


app = FastAPI(
    title="SBFC ITR Verification API",
    description="ITR document detection and validation API",
    version="1.0.0",
)

app.include_router(itr_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
    }