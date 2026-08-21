"""
Main FastAPI Application.

Central entry point for the Document Verification System.

Registered modules:
    1. Bank Statement
    2. Passport
    3. Sale Deed
    4. Driving Licence
    5. Salary Slip
    6. Agent Customer Verification
    7. Agent Property Verification
    8. ITR
    9. PAN
    10. Voter ID
    11. CIBIL
    12. CRIF
"""

from fastapi import FastAPI

from src.api.router import api_router


# ======================================================================
# APPLICATION
# ======================================================================

app = FastAPI(
    title="Document Verification System",
    description=(
        "Modular document detection and verification API."
    ),
    version="2.0.0",
)


# ======================================================================
# MASTER ROUTER
# ======================================================================

app.include_router(
    api_router
)


# ======================================================================
# ROOT
# ======================================================================

@app.get(
    "/",
    tags=["System"],
)
def root():
    return {
        "status": "running",
        "service": "Document Verification System",
        "version": "2.0.0",
    }


# ======================================================================
# HEALTH
# ======================================================================

@app.get(
    "/health",
    tags=["System"],
)
def health():
    return {
        "status": "healthy",
    }