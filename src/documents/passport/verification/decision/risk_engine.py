class RiskEngine:
    """
    Determine verification risk level.
    """

    @staticmethod
    def calculate(
        confidence_result,
        authenticity_result,
        forgery_result
    ):

        confidence = confidence_result["score"]

        authenticity_score = authenticity_result[
            "authenticity"
        ]["score"]

        forgery_score = forgery_result[
            "summary"
        ]["score"]

        # ---------------------------
        # Risk evaluation
        # ---------------------------

        if (

            confidence >= 90

            and

            authenticity_score >= 90

            and

            forgery_score >= 90

        ):

            level = "LOW"

        elif (

            confidence >= 75

            and

            authenticity_score >= 75

            and

            forgery_score >= 75

        ):

            level = "MEDIUM"

        elif (

            confidence >= 50

        ):

            level = "HIGH"

        else:

            level = "CRITICAL"

        return {

            "level": level,

            "confidence_score": confidence,

            "authenticity_score": authenticity_score,

            "forgery_score": forgery_score

        }
