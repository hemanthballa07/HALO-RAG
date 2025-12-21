"""
Verification Controller
Orchestrates claim extraction, entailment verification, and adaptive revision.
This is the control layer that sits between generation and output.
"""

from typing import List, Dict, Any, Callable, Optional
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
        enable_revision: bool = True,
        device: str = None
    ):
        """
        Initialize verification controller.
        
        Args:
            entailment_threshold: Minimum confidence for ENTAILMENT label
            max_revision_cycles: Maximum number of revision attempts
            enable_revision: Whether to enable adaptive revision
            device: Device for models ('cuda' or 'cpu', auto-detected if None)
        """
        self.entailment_threshold = entailment_threshold
        self.max_revision_cycles = max_revision_cycles
        self.enable_revision = enable_revision
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize spaCy claim extractor (Step 2)
        logger.info("Initializing spaCy claim extractor...")
        self.claim_extractor = SpacyClaimExtractor()
        
        # Initialize DeBERTa NLI model (Step 3)
        logger.info("Loading DeBERTa NLI model for entailment verification...")
        model_name = "cross-encoder/nli-deberta-v3-base"
        self.entailment_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.entailment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.entailment_model.eval()
        self.entailment_model.to(self.device)
        
        logger.info(
            f"VerificationController initialized: "
            f"threshold={entailment_threshold}, "
            f"max_cycles={max_revision_cycles}, "
            f"revision_enabled={enable_revision}, "
            f"device={self.device}"
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
        """
        logger.info(f"Verifying answer: '{generated_answer[:100]}...'")
        
        # Step 1: Extract claims (Step 2 - spaCy)
        claims = self._extract_claims(generated_answer)
        logger.info(f"Extracted {len(claims)} claims")
        
        # Step 2: Verify claims (Step 3 - DeBERTa)
        evidence_passages = retrieved_docs.get("passages", [])
        
        verified_claims = []
        for claim in claims:
            verified_claim = self._verify_claim(claim, evidence_passages)
            verified_claims.append(verified_claim)
        
        # Step 3: Determine verification status
        num_verified = sum(1 for c in verified_claims if c.entailment == "ENTAILMENT")
        num_total = len(verified_claims)
        all_verified = num_verified == num_total and num_total > 0
        
        # Compute average confidence
        avg_confidence = sum(c.confidence for c in verified_claims) / max(num_total, 1)
        
        # Generate reason
        if all_verified:
            reason = f"All {num_verified}/{num_total} claims verified"
        elif num_verified > 0:
            reason = f"Only {num_verified}/{num_total} claims verified"
        else:
            reason = f"No claims verified (0/{num_total})"
        
        return VerificationResult(
            claims=verified_claims,
            verified=all_verified,
            revision_cycles=0,
            reason=reason,
            total_claims=num_total,
            verified_claims=num_verified,
            hallucinated_claims=num_total - num_verified,
            avg_confidence=avg_confidence
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
        Verify a single claim against evidence passages using DeBERTa NLI.
        
        Strategy:
        1. For each evidence passage, compute entailment score
        2. Take maximum score across all passages (best evidence)
        3. Assign label based on threshold
        
        Args:
            claim: Claim to verify
            evidence_passages: List of evidence text passages
        
        Returns:
            Updated Claim with entailment label and confidence
        """
        if not evidence_passages:
            claim.entailment = "NEUTRAL"
            claim.confidence = 0.0
            return claim
        
        # Compute entailment scores for each passage
        max_entailment_score = 0.0
        max_contradiction_score = 0.0
        best_passage_idx = 0
        
        for idx, passage in enumerate(evidence_passages):
            # Tokenize claim-passage pair
            inputs = self.entailment_tokenizer(
                claim.text,
                passage,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.entailment_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            # Extract scores (0=contradiction, 1=neutral, 2=entailment)
            probs = probs.cpu().numpy()[0]
            contradiction_score = float(probs[0])
            neutral_score = float(probs[1])
            entailment_score = float(probs[2])
            
            # Track best scores
            if entailment_score > max_entailment_score:
                max_entailment_score = entailment_score
                best_passage_idx = idx
            
            if contradiction_score > max_contradiction_score:
                max_contradiction_score = contradiction_score
        
        # Assign label based on best score
        claim.confidence = max_entailment_score
        
        if max_entailment_score >= self.entailment_threshold:
            claim.entailment = "ENTAILMENT"
            claim.evidence_ids = [f"P{best_passage_idx + 1}"]
        elif max_contradiction_score > 0.7:  # High contradiction threshold
            claim.entailment = "CONTRADICTION"
            claim.evidence_ids = []
        else:
            claim.entailment = "NEUTRAL"
            claim.evidence_ids = []
        
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
