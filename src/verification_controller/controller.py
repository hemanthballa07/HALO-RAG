"""
Verification Controller
Orchestrates claim extraction, entailment verification, and adaptive revision.
This is the control layer that sits between generation and output.
"""

from typing import List, Dict, Any, Callable, Optional
import logging

from .schemas import Claim, VerificationResult, RevisionRequest
from src.verification.spacy_claim_extractor import SpacyClaimExtractor

logger = logging.getLogger(__name__)


class VerificationController:
    """
    Central controller for the verification pipeline.
    
    Flow:
    1. Extract atomic claims from generated text
    2. Verify each claim against retrieved evidence
    3. If verification fails, trigger adaptive revision
    4. Return verified answer OR abstention
    
    This is a STUB implementation. No ML models yet.
    """
    
    def __init__(
        self,
        entailment_threshold: float = 0.75,
        max_revision_cycles: int = 2,
        enable_revision: bool = True
    ):
        """
        Initialize verification controller.
        
        Args:
            entailment_threshold: Minimum confidence for ENTAILMENT label
            max_revision_cycles: Maximum number of revision attempts
            enable_revision: Whether to enable adaptive revision
        """
        self.entailment_threshold = entailment_threshold
        self.max_revision_cycles = max_revision_cycles
        self.enable_revision = enable_revision
        
        # Initialize spaCy claim extractor (NEW - Step 2)
        logger.info("Initializing spaCy claim extractor...")
        self.claim_extractor = SpacyClaimExtractor()
        
        logger.info(
            f"VerificationController initialized: "
            f"threshold={entailment_threshold}, "
            f"max_cycles={max_revision_cycles}, "
            f"revision_enabled={enable_revision}"
        )
    
    def verify(
        self,
        generated_answer: str,
        retrieved_docs: Dict[str, Any],
        query: Optional[str] = None
    ) -> VerificationResult:
        """
        Verify a generated answer against retrieved evidence.
        
        Args:
            generated_answer: The generated text to verify
            retrieved_docs: Dictionary containing retrieved passages and metadata
            query: Optional original query for context
        
        Returns:
            VerificationResult with verification outcome
        
        NOTE: This is a STUB. No actual verification yet.
        """
        logger.info(f"Verifying answer: '{generated_answer[:100]}...'")
        
        # Step 1: Extract claims (stub)
        claims = self._extract_claims(generated_answer)
        logger.info(f"Extracted {len(claims)} claims")
        
        # Step 2: Verify claims (stub)
        verified_claims = []
        for claim in claims:
            # Placeholder: mark all claims as UNVERIFIED
            claim.entailment = "UNVERIFIED"
            claim.confidence = 0.0
            verified_claims.append(claim)
        
        # Step 3: Determine verification status
        num_verified = sum(1 for c in verified_claims if c.entailment == "ENTAILMENT")
        num_total = len(verified_claims)
        all_verified = num_verified == num_total and num_total > 0
        
        return VerificationResult(
            claims=verified_claims,
            verified=all_verified,
            revision_cycles=0,
            reason="Verification not yet implemented - all claims marked UNVERIFIED",
            total_claims=num_total,
            verified_claims=num_verified,
            hallucinated_claims=num_total - num_verified,
            avg_confidence=0.0
        )
    
    def _extract_claims(self, text: str) -> List[Claim]:
        """
        Extract atomic claims from generated text using spaCy SVO extraction.
        
        Args:
            text: Generated text to extract claims from
        
        Returns:
            List of Claim objects
        """
        if not text or text.strip() == "":
            return []
        
        # Use spaCy claim extractor (Step 2 - REAL extraction)
        extracted = self.claim_extractor.extract(text)
        
        # Convert ExtractedClaim objects to Claim objects with IDs
        claims = []
        for idx, extracted_claim in enumerate(extracted, start=1):
            claims.append(
                Claim(
                    claim_id=f"C{idx}",
                    text=extracted_claim.text,
                    evidence_ids=[],
                    entailment="UNVERIFIED",
                    confidence=0.0
                )
            )
        
        return claims
    
    def _verify_claim(
        self,
        claim: Claim,
        evidence_passages: List[str]
    ) -> Claim:
        """
        Verify a single claim against evidence passages.
        
        STUB: Currently does nothing.
        Future: Use DeBERTa-v3 entailment model.
        
        Args:
            claim: Claim to verify
            evidence_passages: List of evidence text passages
        
        Returns:
            Updated Claim with entailment label and confidence
        """
        # Placeholder: return claim unchanged
        return claim
    
    def revise(
        self,
        request: RevisionRequest,
        generation_fn: Callable,
        retrieval_fn: Callable
    ) -> VerificationResult:
        """
        Trigger adaptive revision for failed verification.
        
        STUB: Currently does nothing.
        Future: Implement re-retrieval, constrained generation, claim-by-claim.
        
        Args:
            request: RevisionRequest with failed verification details
            generation_fn: Function to call for regeneration
            retrieval_fn: Function to call for re-retrieval
        
        Returns:
            VerificationResult after revision attempt
        """
        logger.warning("Revision not yet implemented - returning original verification")
        
        # Placeholder: return failed verification
        return VerificationResult(
            claims=request.failed_claims,
            verified=False,
            revision_cycles=0,
            reason="Revision not yet implemented",
            total_claims=len(request.failed_claims),
            verified_claims=0,
            hallucinated_claims=len(request.failed_claims),
            avg_confidence=0.0
        )
