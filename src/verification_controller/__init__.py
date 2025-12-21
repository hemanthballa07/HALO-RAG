"""
Verification Controller Module
Provides structured verification layer for RAG pipeline.
"""

from .controller import VerificationController
from .schemas import Claim, VerificationResult, RevisionRequest

__all__ = [
    "VerificationController",
    "Claim",
    "VerificationResult",
    "RevisionRequest"
]
