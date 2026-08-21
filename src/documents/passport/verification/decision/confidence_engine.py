class ConfidenceEngine:
    """
    Calculate overall confidence based on agreement between
    all verification engines.
    """

    WEIGHTS = {
        "mrz": 30,
        "authenticity": 35,
        "forgery": 35
    }

    @classmethod
    def calculate(
        cls,
        mrz_result,
        authenticity_result,
        forgery_result
    ):

        confidence = 0

        breakdown = {}

        # -------------------------
        # MRZ
        # -------------------------

        if mrz_result.get("passed", False):

            confidence += cls.WEIGHTS["mrz"]

            breakdown["mrz"] = cls.WEIGHTS["mrz"]

        else:

            breakdown["mrz"] = 0

        # -------------------------
        # Authenticity
        # -------------------------

        if authenticity_result.get("passed", False):

            confidence += cls.WEIGHTS["authenticity"]

            breakdown["authenticity"] = cls.WEIGHTS["authenticity"]

        else:

            breakdown["authenticity"] = 0

        # -------------------------
        # Forgery
        # -------------------------

        if forgery_result.get("passed", False):

            confidence += cls.WEIGHTS["forgery"]

            breakdown["forgery"] = cls.WEIGHTS["forgery"]

        else:

            breakdown["forgery"] = 0

        return {

            "confidence": round(
                confidence / 100,
                2
            ),

            "score": confidence,

            "breakdown": breakdown

        }
