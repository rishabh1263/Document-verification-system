"""
Master API Router.

Central router for the Document Verification System.

All document-specific routers are registered here.

Important:
    src.main imports:
        from src.api.router import api_router
"""

from fastapi import APIRouter

router = APIRouter()


# ======================================================================
# BANK STATEMENT
# ======================================================================

from src.documents.bank_statement.api.routes import (
    router as bank_statement_router,
)

router.include_router(
    bank_statement_router,
)


# ======================================================================
# PASSPORT
# ======================================================================

from src.documents.passport.api.upload import (
    router as passport_router,
)

router.include_router(
    passport_router,
    prefix="/passport",
    tags=["Passport"],
)


# ======================================================================
# SALE DEED
# ======================================================================

from src.documents.sale_deed.api.routes import (
    router as sale_deed_router,
)

router.include_router(
    sale_deed_router,
    prefix="/sale-deed",
    tags=["Sale Deed"],
)


# ======================================================================
# DRIVING LICENCE
# ======================================================================

from src.documents.driving_licence.dl_api import (
    router as driving_licence_router,
)

router.include_router(
    driving_licence_router,
    prefix="/driving-licence",
    tags=["Driving Licence"],
)


# ======================================================================
# SALARY SLIP
# ======================================================================

from src.documents.salary_slip.api import (
    router as salary_slip_router,
)

router.include_router(
    salary_slip_router,
    prefix="/salary-slip",
    tags=["Salary Slip"],
)


# ======================================================================
# AGENT CUSTOMER VERIFICATION
# ======================================================================

from src.documents.agent_customer_verification.routes.verify import (
    router as agent_customer_verification_router,
)

router.include_router(
    agent_customer_verification_router,
    prefix="/agent-customer-verification",
    tags=["Agent Customer Verification"],
)


# ======================================================================
# AGENT PROPERTY VERIFICATION
# ======================================================================

from src.documents.agent_property_verification.routes.verification import (
    router as agent_property_verification_router,
)

router.include_router(
    agent_property_verification_router,
    prefix="/agent-property-verification",
    tags=["Agent Property Verification"],
)


# ======================================================================
# ITR
# ======================================================================

from src.documents.itr.api.routes import (
    router as itr_router,
)

router.include_router(
    itr_router,
)


# ======================================================================
# PAN VALIDATION
# ======================================================================

from src.documents.pan.router import (
    router as pan_router,
)

router.include_router(
    pan_router,
)


# ======================================================================
# PAN EXTRACTION
# ======================================================================
#
# Separate from validation.
#
# POST /pan/verify-pan -> validation
# POST /pan/extract    -> extraction
#
# Current extraction field:
#     name
# ======================================================================

from src.documents.pan.extraction_router import (
    router as pan_extraction_router,
)

router.include_router(
    pan_extraction_router,
)


# ======================================================================
# VOTER ID
# ======================================================================

from src.documents.voter_id.router import (
    router as voter_id_router,
)

router.include_router(
    voter_id_router,
)


# ======================================================================
# CIBIL
# ======================================================================

from src.documents.cibil.router import (
    router as cibil_router,
)

router.include_router(
    cibil_router,
)


# ======================================================================
# CRIF
# ======================================================================

from src.documents.crif.router import (
    router as crif_router,
)

router.include_router(
    crif_router,
)


# ======================================================================
# COMPATIBILITY ALIAS
# ======================================================================

api_router = router