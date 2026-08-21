from uuid import uuid4


def generate_request_id() -> str:
    return f"REQ-{uuid4().hex[:12].upper()}"
