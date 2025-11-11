"""
Verification module for entailment-based factual verification.
"""

from .entailment_verifier import EntailmentVerifier
from .claim_extractor import ClaimExtractor
from .lexical_verifier import LexicalOverlapVerifier

__all__ = ["EntailmentVerifier", "ClaimExtractor", "LexicalOverlapVerifier"]

