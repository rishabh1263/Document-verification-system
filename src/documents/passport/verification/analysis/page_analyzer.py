class PageAnalyzer:

    @classmethod
    def analyze(
        cls,
        page_number,
        layout,
        mrz,
        document
    ):

        score = 0

        reasons = []

        if document["document_type"] == "PASSPORT":

            score += 50

            reasons.append(
                "Passport document detected."
            )

        else:

            reasons.append(
                "Document is not classified as a passport."
            )

        if mrz["passed"]:

            score += 40

            reasons.append(
                "MRZ-like region detected."
            )

        else:

            reasons.append(
                "MRZ region not detected."
            )

        score += int(document["confidence"] * 10)

        return {

            "page": page_number,

            "score": score,

            "candidate": score >= 50,

            "reasons": reasons
        }
