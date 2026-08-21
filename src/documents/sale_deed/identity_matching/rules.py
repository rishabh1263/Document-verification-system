"""
rules.py

Configuration-driven document comparison rules.

This module defines which fields should be compared between
different document types.

Each rule contains:

field_name   -> Name displayed in report
source_field -> Field in source document
target_field -> Field in target document
matcher      -> Which matcher to use
required     -> Whether field is mandatory
weight       -> Contribution to overall confidence score
"""

# ==========================================================
# MATCHER TYPES
# ==========================================================

MATCHER_NAME = "name"
MATCHER_ADDRESS = "address"
MATCHER_TEXT = "text"
MATCHER_DATE = "date"
MATCHER_DOCUMENT = "document_number"


# ==========================================================
# SALE DEED â†” AADHAAR (BUYER)
# ==========================================================

SALE_DEED_BUYER_AADHAAR_RULES = [

    {
        "field_name": "Buyer Name",
        "source_field": "buyer",
        "target_field": "name",
        "matcher": MATCHER_NAME,
        "required": True,
        "weight": 40
    },

    {
        "field_name": "Father Name",
        "source_field": "father_name",
        "target_field": "father_name",
        "matcher": MATCHER_NAME,
        "required": False,
        "weight": 20
    },

    {
        "field_name": "Address",
        "source_field": "address",
        "target_field": "address",
        "matcher": MATCHER_ADDRESS,
        "required": False,
        "weight": 25
    },

    {
        "field_name": "Aadhaar Number",
        "source_field": "aadhaar_number",
        "target_field": "aadhaar_number",
        "matcher": MATCHER_DOCUMENT,
        "required": False,
        "weight": 15
    }

]


# ==========================================================
# SALE DEED â†” AADHAAR (SELLER)
# ==========================================================

SALE_DEED_SELLER_AADHAAR_RULES = [

    {
        "field_name": "Seller Name",
        "source_field": "seller",
        "target_field": "name",
        "matcher": MATCHER_NAME,
        "required": True,
        "weight": 40
    },

    {
        "field_name": "Father Name",
        "source_field": "father_name",
        "target_field": "father_name",
        "matcher": MATCHER_NAME,
        "required": False,
        "weight": 20
    },

    {
        "field_name": "Address",
        "source_field": "address",
        "target_field": "address",
        "matcher": MATCHER_ADDRESS,
        "required": False,
        "weight": 25
    },

    {
        "field_name": "Aadhaar Number",
        "source_field": "aadhaar_number",
        "target_field": "aadhaar_number",
        "matcher": MATCHER_DOCUMENT,
        "required": False,
        "weight": 15
    }

]


# ==========================================================
# SALE DEED â†” PAN (BUYER)
# ==========================================================

SALE_DEED_BUYER_PAN_RULES = [

    {
        "field_name": "Buyer Name",
        "source_field": "buyer",
        "target_field": "name",
        "matcher": MATCHER_NAME,
        "required": True,
        "weight": 60
    },

    {
        "field_name": "PAN Number",
        "source_field": "pan_number",
        "target_field": "pan_number",
        "matcher": MATCHER_DOCUMENT,
        "required": True,
        "weight": 40
    }

]


# ==========================================================
# SALE DEED â†” PASSPORT (BUYER)
# ==========================================================

SALE_DEED_BUYER_PASSPORT_RULES = [

    {
        "field_name": "Buyer Name",
        "source_field": "buyer",
        "target_field": "name",
        "matcher": MATCHER_NAME,
        "required": True,
        "weight": 35
    },

    {
        "field_name": "Passport Number",
        "source_field": "passport_number",
        "target_field": "passport_number",
        "matcher": MATCHER_DOCUMENT,
        "required": True,
        "weight": 35
    },

    {
        "field_name": "Address",
        "source_field": "address",
        "target_field": "address",
        "matcher": MATCHER_ADDRESS,
        "required": False,
        "weight": 30
    }

]


# ==========================================================
# SALE DEED â†” DRIVING LICENSE (BUYER)
# ==========================================================

SALE_DEED_BUYER_DL_RULES = [

    {
        "field_name": "Buyer Name",
        "source_field": "buyer",
        "target_field": "name",
        "matcher": MATCHER_NAME,
        "required": True,
        "weight": 40
    },

    {
        "field_name": "License Number",
        "source_field": "license_number",
        "target_field": "license_number",
        "matcher": MATCHER_DOCUMENT,
        "required": True,
        "weight": 35
    },

    {
        "field_name": "Address",
        "source_field": "address",
        "target_field": "address",
        "matcher": MATCHER_ADDRESS,
        "required": False,
        "weight": 25
    }

]


# ==========================================================
# SALE DEED â†” VOTER ID (BUYER)
# ==========================================================

SALE_DEED_BUYER_VOTER_RULES = [

    {
        "field_name": "Buyer Name",
        "source_field": "buyer",
        "target_field": "name",
        "matcher": MATCHER_NAME,
        "required": True,
        "weight": 45
    },

    {
        "field_name": "Voter ID Number",
        "source_field": "voter_id_number",
        "target_field": "voter_id_number",
        "matcher": MATCHER_DOCUMENT,
        "required": True,
        "weight": 30
    },

    {
        "field_name": "Address",
        "source_field": "address",
        "target_field": "address",
        "matcher": MATCHER_ADDRESS,
        "required": False,
        "weight": 25
    }

]


# ==========================================================
# RULE REGISTRY
# ==========================================================

DOCUMENT_COMPARISON_RULES = {

    ("sale_deed", "aadhaar", "buyer"):
        SALE_DEED_BUYER_AADHAAR_RULES,

    ("sale_deed", "aadhaar", "seller"):
        SALE_DEED_SELLER_AADHAAR_RULES,

    ("sale_deed", "pan", "buyer"):
        SALE_DEED_BUYER_PAN_RULES,

    ("sale_deed", "passport", "buyer"):
        SALE_DEED_BUYER_PASSPORT_RULES,

    ("sale_deed", "driving_license", "buyer"):
        SALE_DEED_BUYER_DL_RULES,

    ("sale_deed", "voter_id", "buyer"):
        SALE_DEED_BUYER_VOTER_RULES,

}


# ==========================================================
# HELPER
# ==========================================================

def get_rules(
    source_document: str,
    target_document: str,
    role: str = "buyer"
):
    """
    Returns comparison rules for the given documents.

    Example:
        get_rules(
            "sale_deed",
            "aadhaar",
            "buyer"
        )
    """

    return DOCUMENT_COMPARISON_RULES.get(
        (
            source_document.lower(),
            target_document.lower(),
            role.lower()
        ),
        []
    )
