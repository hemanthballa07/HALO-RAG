"""
Adaptive Revision Strategies Module
Implements re-retrieval, constrained generation, and claim-by-claim regeneration.
"""

from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


class RevisionStrategy(Enum):
    """Revision strategy types."""
    RE_RETRIEVAL = "re_retrieval"
    CONSTRAINED_GENERATION = "constrained_generation"
    CLAIM_BY_CLAIM = "claim_by_claim"


class AdaptiveRevisionStrategy:
    """
    Adaptive revision strategies for hallucination reduction.
    """
    
    def __init__(
        self,
        max_iterations: int = 3,
        strategies: List[str] = None
    ):
        """
        Initialize adaptive revision strategy.
        
        Args:
            max_iterations: Maximum revision iterations
            strategies: List of strategy names to use
        """
        self.max_iterations = max_iterations
        
        if strategies is None:
            strategies = [
                "re_retrieval",
                "constrained_generation",
                "claim_by_claim"
            ]
        
        self.strategies = strategies
    
    def revise(
        self,
        query: str,
        initial_generation: str,
        verification_results: Dict[str, Any],
        retrieval_fn,
        generation_fn,
        verification_fn,
        claim_extractor_fn,
        iteration: int = 0
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Apply adaptive revision strategies.
        
        Args:
            query: Original query
            initial_generation: Initial generated text
            verification_results: Verification results from first pass
            retrieval_fn: Function to retrieve documents
            generation_fn: Function to generate text
            verification_fn: Function to verify claims
            claim_extractor_fn: Function to extract claims from text
            iteration: Current iteration number
        
        Returns:
            Tuple of (revised_generation, new_verification_results, strategy_metadata)
            strategy_metadata contains: strategy_name, prompt_used, expanded_query (if applicable),
            verified_claims (if applicable), claim_queries (if applicable)
        """
        if iteration >= self.max_iterations:
            return initial_generation, verification_results, {"strategy_name": "none", "prompt_used": None}
        
        # Check if verification passed
        if verification_results.get("verified", False):
            return initial_generation, verification_results, {"strategy_name": "none", "prompt_used": None}
        
        # Determine which strategy to use
        strategy = self._select_strategy(verification_results, iteration)
        
        if strategy == RevisionStrategy.RE_RETRIEVAL:
            return self._re_retrieval_strategy(
                query,
                initial_generation,
                verification_results,
                retrieval_fn,
                generation_fn,
                verification_fn,
                claim_extractor_fn,
                iteration
            )
        elif strategy == RevisionStrategy.CONSTRAINED_GENERATION:
            return self._constrained_generation_strategy(
                query,
                initial_generation,
                verification_results,
                retrieval_fn,
                generation_fn,
                verification_fn,
                claim_extractor_fn,
                iteration
            )
        elif strategy == RevisionStrategy.CLAIM_BY_CLAIM:
            return self._claim_by_claim_strategy(
                query,
                initial_generation,
                verification_results,
                retrieval_fn,
                generation_fn,
                verification_fn,
                claim_extractor_fn,
                iteration
            )
        else:
            return initial_generation, verification_results, {"strategy_name": "none", "prompt_used": None}
    
    def _select_strategy(
        self,
        verification_results: Dict[str, Any],
        iteration: int
    ) -> RevisionStrategy:
        """Select revision strategy based on verification results."""
        entailment_rate = verification_results.get("entailment_rate", 0.0)
        
        # Low entailment rate: try re-retrieval first
        if entailment_rate < 0.5 and "re_retrieval" in self.strategies:
            return RevisionStrategy.RE_RETRIEVAL
        
        # Medium entailment rate: use constrained generation
        elif entailment_rate < 0.8 and "constrained_generation" in self.strategies:
            return RevisionStrategy.CONSTRAINED_GENERATION
        
        # High but not perfect: use claim-by-claim
        elif "claim_by_claim" in self.strategies:
            return RevisionStrategy.CLAIM_BY_CLAIM
        
        return RevisionStrategy.RE_RETRIEVAL
    
    def _re_retrieval_strategy(
        self,
        query: str,
        initial_generation: str,
        verification_results: Dict[str, Any],
        retrieval_fn,
        generation_fn,
        verification_fn,
        claim_extractor_fn,
        iteration: int
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Re-retrieval strategy: expand query and retrieve more documents.
        """
        # Expand query with failed claims
        failed_claims = [
            r["claim"] for r in verification_results.get("verification_results", [])
            if not r.get("is_entailed", False)
        ]
        
        if failed_claims:
            expanded_query = f"{query} {' '.join(failed_claims[:2])}"
        else:
            expanded_query = query
        
        # Re-retrieve with expanded query
        new_contexts = retrieval_fn(expanded_query, top_k=20)
        
        # Re-generate with new contexts
        context_text = " ".join([ctx[1] for ctx in new_contexts])
        revised_generation = generation_fn(query, context_text)
        
        # Construct the actual prompt that was used
        prompt_used = f"Question: {query} Context: {context_text[:200]}... Answer:"
        
        # Extract claims from revised generation (not from old verification results)
        revised_claims = claim_extractor_fn(revised_generation)
        
        # Re-verify with extracted claims
        new_verification = verification_fn(
            revised_generation,
            [ctx[1] for ctx in new_contexts],
            revised_claims
        )
        
        # Strategy metadata
        strategy_metadata = {
            "strategy_name": "re_retrieval",
            "prompt_used": prompt_used,
            "expanded_query": expanded_query,
            "original_query": query,
            "failed_claims_used": failed_claims[:2] if failed_claims else []
        }
        
        return revised_generation, new_verification, strategy_metadata
    
    def _constrained_generation_strategy(
        self,
        query: str,
        initial_generation: str,
        verification_results: Dict[str, Any],
        retrieval_fn,
        generation_fn,
        verification_fn,
        claim_extractor_fn,
        iteration: int
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Constrained generation: generate with verified claims as constraints.
        """
        # Get verified claims
        verified_claims = [
            r["claim"] for r in verification_results.get("verification_results", [])
            if r.get("is_entailed", False)
        ]
        
        # Get contexts
        contexts = retrieval_fn(query, top_k=10)
        context_text = " ".join([ctx[1] for ctx in contexts])
        
        # Generate with verified claims as constraints
        # The generator now supports verified_claims parameter
        revised_generation = generation_fn(
            query,
            context_text,
            verified_claims=verified_claims
        )
        
        # Construct the actual prompt that was used (with verified claims)
        if verified_claims:
            verified_text = " Verified facts that must be included: " + " | ".join(verified_claims)
            full_context = context_text + verified_text
        else:
            full_context = context_text
        prompt_used = f"Question: {query} Context: {full_context[:200]}... Answer:"
        
        # Extract claims from revised generation
        revised_claims = claim_extractor_fn(revised_generation)
        
        # Re-verify with extracted claims
        new_verification = verification_fn(
            revised_generation,
            [ctx[1] for ctx in contexts],
            revised_claims
        )
        
        # Strategy metadata
        strategy_metadata = {
            "strategy_name": "constrained_generation",
            "prompt_used": prompt_used,
            "verified_claims": verified_claims,
            "original_query": query
        }
        
        return revised_generation, new_verification, strategy_metadata
    
    def _claim_by_claim_strategy(
        self,
        query: str,
        initial_generation: str,
        verification_results: Dict[str, Any],
        retrieval_fn,
        generation_fn,
        verification_fn,
        claim_extractor_fn,
        iteration: int
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Claim-by-claim strategy: regenerate only unverified claims.
        """
        # Get unverified claims
        unverified_claims = [
            r["claim"] for r in verification_results.get("verification_results", [])
            if not r.get("is_entailed", False)
        ]
        
        if not unverified_claims:
            return initial_generation, verification_results, {"strategy_name": "claim_by_claim", "prompt_used": None}
        
        # Get verified claims
        verified_claims = [
            r["claim"] for r in verification_results.get("verification_results", [])
            if r.get("is_entailed", False)
        ]
        
        # Retrieve contexts for unverified claims
        contexts = retrieval_fn(query, top_k=10)
        context_text = " ".join([ctx[1] for ctx in contexts])
        
        # Regenerate only unverified claims
        # For each unverified claim, generate a replacement
        revised_claims = []
        claim_queries = []  # Track individual queries used
        claim_prompts = []  # Track prompts used for each claim
        for claim in unverified_claims:
            # Generate replacement claim
            claim_query = f"{query} Specifically about: {claim}"
            claim_queries.append(claim_query)
            
            try:
                replacement = generation_fn(claim_query, context_text, max_new_tokens=64)
            except TypeError:
                # Generator doesn't support max_new_tokens
                replacement = generation_fn(claim_query, context_text)
            
            revised_claims.append(replacement)
            
            # Construct prompt for this claim
            claim_prompt = f"Question: {claim_query} Context: {context_text[:200]}... Answer:"
            claim_prompts.append(claim_prompt)
        
        # Reconstruct generation with verified + revised claims
        revised_generation = " ".join(verified_claims + revised_claims)
        
        # Extract claims from revised generation
        extracted_claims = claim_extractor_fn(revised_generation)
        
        # Re-verify with extracted claims
        new_verification = verification_fn(
            revised_generation,
            [ctx[1] for ctx in contexts],
            extracted_claims
        )
        
        # Strategy metadata
        strategy_metadata = {
            "strategy_name": "claim_by_claim",
            "prompt_used": claim_prompts,  # List of prompts, one per unverified claim
            "claim_queries": claim_queries,  # List of focused queries used
            "unverified_claims": unverified_claims,
            "verified_claims": verified_claims,
            "original_query": query
        }
        
        return revised_generation, new_verification, strategy_metadata

