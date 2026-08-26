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
    13. Signature Verification
"""

from __future__ import annotations


# ======================================================================
# SSL BOOTSTRAP
# ======================================================================
#
# Must run BEFORE importing the master API router.
#
# Some imported services initialize external ML models during startup.
# truststore allows Python to use the Windows/system certificate store.
# ======================================================================

from src.ssl_bootstrap import initialize_ssl


initialize_ssl()


# ======================================================================
# FASTAPI
# ======================================================================

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
    api_router,
)


# ======================================================================
# ROOT
# ======================================================================

@app.get(
    "/",
    tags=["System"],
)
def root() -> dict[str, str]:

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
def health() -> dict[str, str]:

    return {
        "status": "healthy",
    }