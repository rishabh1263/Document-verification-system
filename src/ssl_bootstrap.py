"""
Application SSL bootstrap.

Uses the Windows/system certificate store through truststore so
Python HTTPS clients can trust corporate CA certificates.
"""

import truststore


def initialize_ssl() -> None:
    """
    Inject the system certificate store into Python's SSL handling.

    This must run before importing modules that create HTTPS
    connections during application startup.
    """
    truststore.inject_into_ssl()