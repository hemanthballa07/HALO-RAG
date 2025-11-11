"""
Simple lexical overlap verifier for ablation study.
Uses token overlap instead of NLI-based verification.
"""

from typing import List, Dict, Any
import re


class LexicalOverlapVerifier:
    """
    Simple lexical overlap verifier.
    Verifies claims by checking token overlap with context.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize lexical overlap verifier.
        
        Args:
            threshold: Minimum token overlap ratio for entailment
        """
        self.threshold = threshold
    
    def _tokenize(self, text: str) -> set:
        """Tokenize text into set of lowercase words."""
        if not text:
            return set()
        # Simple tokenization: lowercase, split on whitespace/punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return set(tokens)
    
    def verify_claim(
        self,
        claim: str,
        context: str
    ) -> Dict[str, float]:
        """
        Verify a single claim against context using token overlap.
        
        Args:
            claim: Claim to verify
            context: Context to verify against
        
        Returns:
            Dictionary with 'entailment', 'neutral', 'contradiction' scores
            (lexical overlap is used as entailment score)
        """
        claim_tokens = self._tokenize(claim)
        context_tokens = self._tokenize(context)
        
        if len(claim_tokens) == 0:
            return {
                "contradiction": 0.0,
                "neutral": 1.0,
                "entailment": 0.0
            }
        
        # Compute token overlap ratio
        overlap = claim_tokens & context_tokens
        overlap_ratio = len(overlap) / len(claim_tokens) if len(claim_tokens) > 0 else 0.0
        
        # Use overlap ratio as entailment score
        entailment_score = overlap_ratio
        
        # Simple heuristics for contradiction and neutral
        # If overlap is very low, likely contradiction or no evidence
        if overlap_ratio < 0.2:
            contradiction_score = 0.5
            neutral_score = 0.5
            entailment_score = 0.0
        else:
            contradiction_score = max(0.0, (1.0 - overlap_ratio) * 0.3)
            neutral_score = max(0.0, (1.0 - overlap_ratio) * 0.2)
        
        # Normalize to sum to 1.0
        total = entailment_score + contradiction_score + neutral_score
        if total > 0:
            entailment_score = entailment_score / total
            contradiction_score = contradiction_score / total
            neutral_score = neutral_score / total
        else:
            entailment_score = 0.0
            contradiction_score = 0.0
            neutral_score = 1.0
        
        return {
            "contradiction": contradiction_score,
            "neutral": neutral_score,
            "entailment": entailment_score
        }
    
    def is_entailed(
        self,
        claim: str,
        context: str,
        threshold: float = None
    ) -> tuple:
        """
        Check if claim is entailed by context (above threshold).
        
        Args:
            claim: Claim to verify
            context: Context to verify against
            threshold: Entailment threshold (default: self.threshold)
        
        Returns:
            Tuple of (is_entailed: bool, entailment_score: float)
        """
        if threshold is None:
            threshold = self.threshold
        
        result = self.verify_claim(claim, context)
        entailment_score = result["entailment"]
        is_entailed = entailment_score >= threshold
        
        return is_entailed, entailment_score
    
    def verify_generation(
        self,
        generated_text: str,
        retrieved_contexts: List[str],
        claims: List[str]
    ) -> Dict[str, Any]:
        """
        Verify all claims in generated text against retrieved contexts.
        
        Args:
            generated_text: Generated text
            retrieved_contexts: List of retrieved context documents
            claims: List of extracted claims
        
        Returns:
            Dictionary with verification results (same format as EntailmentVerifier)
        """
        # Combine contexts for verification
        combined_context = " ".join(retrieved_contexts[:3])  # Use top 3 contexts
        
        verification_results = []
        for claim in claims:
            # Get full verification result
            full_result = self.verify_claim(claim, combined_context)
            is_entailed, score = self.is_entailed(claim, combined_context)
            
            # Determine label based on scores
            entailment_score = full_result["entailment"]
            contradiction_score = full_result["contradiction"]
            neutral_score = full_result["neutral"]
            
            # Label: highest probability wins
            if entailment_score >= self.threshold:
                label = "ENTAILED"
            elif contradiction_score > neutral_score and contradiction_score > 0.5:
                label = "CONTRADICTED"
            else:
                label = "NO_EVIDENCE"
            
            verification_results.append({
                "claim": claim,
                "is_entailed": is_entailed,
                "entailment_score": score,
                "contradiction_score": contradiction_score,
                "neutral_score": neutral_score,
                "label": label,
                "threshold": self.threshold
            })
        
        # Compute aggregate metrics
        num_entailed = sum(1 for r in verification_results if r["is_entailed"])
        num_total = len(verification_results)
        entailment_rate = num_entailed / num_total if num_total > 0 else 0.0
        avg_score = sum(r["entailment_score"] for r in verification_results) / num_total if num_total > 0 else 0.0
        
        return {
            "verification_results": verification_results,
            "num_entailed": num_entailed,
            "num_total": num_total,
            "entailment_rate": entailment_rate,
            "avg_entailment_score": avg_score,
            "verified": entailment_rate >= 0.90  # 90% claims must be entailed
        }
    
    def set_threshold(self, threshold: float):
        """Update entailment threshold."""
        self.threshold = threshold

