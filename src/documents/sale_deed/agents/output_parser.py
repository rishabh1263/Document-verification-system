import json
import re


class OutputParser:

    @staticmethod
    def parse(response: str):

        response = response.strip()

        response = re.sub(r"^```json", "", response)
        response = re.sub(r"^```", "", response)
        response = re.sub(r"```$", "", response)

        response = response.strip()

        try:
            return json.loads(response)

        except Exception:
            return None
