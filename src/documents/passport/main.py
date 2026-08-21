from fastapi import FastAPI

from src.documents.passport.core.config import settings
from src.documents.passport.core.logger import logger
from src.documents.passport.api.upload import router as upload_router


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION
)
app.include_router(
    upload_router,
    prefix="/api/v1",
    tags=["Upload"]
)

@app.on_event("startup")
async def startup():

    logger.info("===================================")
    logger.info("Passport Verification API Started")
    logger.info("===================================")


@app.get("/")
async def root():
    return {
        "application": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "Running"
    }


@app.get("/health")
async def health():
    return {
        "status": "Healthy"
    }
