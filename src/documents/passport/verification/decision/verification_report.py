class VerificationReport:
    """
    Generates a clean verification response for the API.
    """

    @classmethod
    def generate(
        cls,
        request_id,
        document_type,
        authenticity_result,
        forgery_result,
        decision_result
    ):

        issues = []

        for message in decision_result.get("warnings", []):

            msg = message.lower()

            # Keep only actual issues
            if (
                "not detected" in msg
                or "failed" in msg
                or "invalid" in msg
                or "mismatch" in msg
                or "inconsistent" in msg
                or "forgery" in msg
                or "tamper" in msg
                or "suspicious" in msg
            ):
                issues.append(message)

        return {

            "request_id": request_id,

            "document_type": document_type,

            "status": (
                "VERIFIED"
                if decision_result.get("passed", False)
                else "FAILED"
            ),

            "decision": decision_result["decision"],

            "confidence": round(
                decision_result["confidence"],
                2
            ),

            "authenticity_score": round(
                authenticity_result["authenticity"]["score"],
                2
            ),

            "forgery_score": round(
                forgery_result["summary"]["score"],
                2
            ),

            "issues": issues

        }
