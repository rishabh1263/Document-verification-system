VERIFICATION_PROMPT = """
You are an expert in Indian Property Registration Documents.

Determine whether the OCR text belongs to an Indian Sale Deed.

OCR TEXT
==================================================

__DOCUMENT_TEXT__

==================================================

A Sale Deed typically contains several of these indicators:

â€¢ Sale Deed / à¤µà¤¿à¤•à¥à¤°à¤¯ à¤µà¤¿à¤²à¥‡à¤– / Conveyance Deed
â€¢ Registration Office / Sub Registrar
â€¢ Registration Date
â€¢ Registration Number
â€¢ Book Number
â€¢ Volume Number
â€¢ Page Number
â€¢ Token Number
â€¢ Executed By / Executant
â€¢ In Favour Of
â€¢ Vendor / Seller
â€¢ Buyer / Purchaser
â€¢ Stamp Duty
â€¢ Registration Fee
â€¢ Survey Number / Plot Number / Khata Number / Khasra Number

Rules

- Use only the OCR text.
- Do not guess.
- If enough Sale Deed indicators are present, verify the document.
- If the document is another document type, set verified=false.
- If the OCR is unreadable or insufficient, set manual_review_required=true.

Return ONLY valid JSON.

{
    "verified": true,
    "confidence": 95,
    "document_type": "Sale Deed",
    "reason": "",
    "manual_review_required": false
}
"""

EXTRACTION_PROMPT = '''
You are a legal document extraction engine. Extract structured data from the following Indian Sale Deed OCR text.
Return ONLY a valid JSON object. Do not include markdown formatting or explanations.

## INDIAN SALE DEED CONVENTIONS (STRICT):
1. "Executed By" = SELLERS/VENDORS/TRANSFERORS/EXECUTANTS (people signing AWAY the property)
2. "In Favour Of" = BUYER/PURCHASER/VENDEE/TRANSFEREE (person ACQUIRING the property)
3. "Presented By" = Person who physically presented the document at the registration office
4. Sale Consideration = The total sale price. Search aggressively for: "à¤®à¥‚à¤²à¥à¤¯", "à¤¬à¤¿à¤•à¥à¤°à¥€ à¤®à¥‚à¤²à¥à¤¯", "consideration", "sale price", "à¤°à¥à¤ªà¤¯à¥‡", "Rs.", amounts near "sold" or "bought"
5. Registration Fee = Search for: "à¤ªà¤‚à¤œà¥€à¤•à¤°à¤£ à¤¶à¥à¤²à¥à¤•", "registration fee", "reg. fee", "registration charges"
6. Property Details = Look in the "Schedule of Property", "à¤µà¤¿à¤µà¤°à¤£", or "Details of Property" section for: district, village, tehsil, khasra, khata, survey number, plot number, area, boundaries
7. Registration Number = Look for: "à¤ªà¤‚à¤œà¥€à¤•à¤°à¤£ à¤¸à¤‚à¤–à¥à¤¯à¤¾", "registration no", "reg. no."
8. Serial Number = Look for: "à¤•à¥à¤°à¤® à¤¸à¤‚à¤–à¥à¤¯à¤¾", "serial no", "s.no"
9. If a field is genuinely not found, use empty string "" or empty array [] â€” never null

## PARTY ROLE MAPPING (MANDATORY):
- Populate ALL synonym arrays with the SAME names:
  - vendor, seller, transferor, executant  =  executed_by
  - buyer, purchaser, vendee, transferee   =  in_favour_of
  - presenter                                =  presented_by
- The same person MAY appear under multiple roles.
- Do NOT leave buyer[], seller[], vendor[], transferor[] empty if executed_by / in_favour_of are present.

## EXAMPLES OF SALE CONSIDERATION EXTRACTION:
- "à¤¬à¤¿à¤•à¥à¤°à¥€ à¤®à¥‚à¤²à¥à¤¯ à¤°à¥à¤ªà¤¯à¥‡ 5,00,000" â†’ "Rs. 500000"
- "Sale Consideration: Rs. 1,50,000 (Rupees One Lakh Fifty Thousand only)" â†’ "Rs. 150000"
- "à¤®à¥‚à¤²à¥à¤¯ à¤°à¤¾à¤¶à¤¿ à¤°à¥à¥¦ à¥¨à¥«,à¥¦à¥¦à¥¦" â†’ "Rs. 25000"
- "consideration amount of Rs. 3,20,000" â†’ "Rs. 320000"
- "à¤¬à¤¿à¤•à¥à¤°à¤¯ à¤µà¤¿à¤²à¥‡à¤– à¤•à¥€ à¤•à¥à¤² à¤°à¤¾à¤¶à¤¿ à¤°à¥à¤ªà¤¯à¥‡ 10,00,000" â†’ "Rs. 1000000"
- "that I have sold... for a sum of Rs. 7,50,000" â†’ "Rs. 750000"
- "à¤•à¥à¤² à¤®à¥‚à¤²à¥à¤¯ à¤°à¥. 450000" â†’ "Rs. 450000"

## PROPERTY DETAILS EXTRACTION:
The property schedule is usually on pages 3-6. Look for sections titled:
- "Schedule of Property"
- "Property Details"
- "à¤µà¤¿à¤µà¤°à¤£"
- "Details of Land"
- "à¤œà¤®à¥€à¤¨ à¤•à¤¾ à¤µà¤¿à¤µà¤°à¤£"
- "Schedule"
- "Description of Property"

Extract EVERY field mentioned, even if partially visible.

## OUTPUT JSON SCHEMA:
{
  "document_details": {
    "deed_number": "",
    "token_number": "",
    "registration_date": "",
    "registration_office": "",
    "book_number": "",
    "volume_number": "",
    "page_number": "",
    "registration_number": "",
    "serial_number": "",
    "document_type": "Sale Deed"
  },
  "party_roles": {
    "in_favour_of": [],
    "executed_by": [],
    "presented_by": [],
    "identifier": [],
    "witness": [],
    "vendor": [],
    "seller": [],
    "buyer": [],
    "purchaser": [],
    "vendee": [],
    "transferor": [],
    "transferee": [],
    "executant": [],
    "presenter": [],
    "claimant": []
  },
  "seller": [],
  "buyer": [],
  "property": {
    "district": "",
    "village": "",
    "survey_number": "",
    "plot_number": "",
    "area": "",
    "boundary": "",
    "sub_district": "",
    "taluka": "",
    "tehsil": "",
    "khasra_number": "",
    "khata_number": ""
  },
  "financial": {
    "stamp_duty": "",
    "registration_fee": "",
    "sale_consideration": "",
    "market_value": "",
    "other_fee": ""
  }
}

## OCR TEXT:
__DOCUMENT_TEXT__
'''

FOCUSED_EXTRACTION_PROMPT = '''
You are a legal document extraction engine. Extract ONLY the __FOCUS__ fields from this section of an Indian Sale Deed.
Return ONLY a valid JSON object.

## SECTION TEXT:
__SECTION_TEXT__

## INSTRUCTIONS:
- Search aggressively for every field listed below.
- If a field is not found, use empty string "".
- For amounts, include "Rs." prefix if present.

## OUTPUT:
{
  "document_details": {
    "registration_number": "",
    "serial_number": "",
    "registration_date": ""
  },
  "financial": {
    "stamp_duty": "",
    "registration_fee": "",
    "sale_consideration": "",
    "market_value": "",
    "other_fee": ""
  },
  "property": {
    "district": "",
    "village": "",
    "survey_number": "",
    "plot_number": "",
    "area": "",
    "boundary": "",
    "sub_district": "",
    "taluka": "",
    "tehsil": "",
    "khasra_number": "",
    "khata_number": ""
  }
}
'''
