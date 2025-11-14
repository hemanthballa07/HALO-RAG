"""
End-to-end Self-Verification RAG Pipeline
Combines retrieval, reranking, generation, and verification.
"""

from typing import List, Dict, Optional, Tuple, Any
import torch
import logging

from src.retrieval import HybridRetriever, CrossEncoderReranker
from src.generator import FLANT5Generator
from src.verification import EntailmentVerifier, ClaimExtractor
from src.revision import AdaptiveRevisionStrategy
from src.evaluation import EvaluationMetrics

logger = logging.getLogger(__name__)


class SelfVerificationRAGPipeline:
    """
    End-to-end Self-Verification RAG Pipeline.
    
    Components:
    1. Hybrid retrieval (FAISS + BM25)
    2. Cross-encoder reranking
    3. FLAN-T5 generation (with QLoRA)
    4. Entailment-based verification
    5. Adaptive revision (if needed)
    """
    
    def __init__(
        self,
        corpus: List[str],
        retrieval_model: str = "sentence-transformers/all-mpnet-base-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        generator_model: str = "google/flan-t5-large",
        verifier_model: str = "cross-encoder/nli-deberta-v3-base",
        entailment_threshold: float = 0.75,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        device: str = "cuda",
        use_qlora: bool = True,
        generator_lora_checkpoint: Optional[str] = None,
        enable_revision: bool = True,
        max_revision_iterations: int = 3,
        **kwargs
    ):
        """
        Initialize Self-Verification RAG pipeline.
        
        Args:
            corpus: List of documents to retrieve from
            retrieval_model: Dense retrieval model name
            reranker_model: Cross-encoder reranker model name
            generator_model: Generator model name
            verifier_model: Entailment verifier model name
            entailment_threshold: Entailment threshold (τ)
            dense_weight: Weight for dense retrieval
            sparse_weight: Weight for sparse retrieval
            device: Device to run models on
            use_qlora: Whether to use QLoRA for generator
            enable_revision: Whether to enable adaptive revision
            max_revision_iterations: Maximum revision iterations
        """
        self.device = device
        self.corpus = corpus
        self.enable_revision = enable_revision
        self.max_revision_iterations = max_revision_iterations
        
        # Initialize retrieval
        logger.info("Initializing hybrid retriever...")
        self.retriever = HybridRetriever(
            dense_model_name=retrieval_model,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            device=device
        )
        self.retriever.build_index(corpus)
        
        # Initialize reranker
        logger.info("Initializing cross-encoder reranker...")
        self.reranker = CrossEncoderReranker(
            model_name=reranker_model,
            device=device
        )
        
        # Initialize generator
        logger.info("Initializing FLAN-T5 generator...")
        self.generator = FLANT5Generator(
            model_name=generator_model,
            device=device,
            lora_checkpoint=generator_lora_checkpoint,
            use_qlora=use_qlora
        )
        
        # Initialize verifier
        logger.info("Initializing entailment verifier...")
        self.verifier = EntailmentVerifier(
            model_name=verifier_model,
            device=device,
            threshold=entailment_threshold
        )
        
        # Initialize claim extractor
        logger.info("Initializing claim extractor...")
        self.claim_extractor = ClaimExtractor()
        
        # Initialize revision strategy
        if enable_revision:
            logger.info("Initializing adaptive revision strategy...")
            # Get revision config from kwargs if provided, otherwise use defaults
            revision_config = kwargs.get("revision_config", {})
            strategy_selection_mode = revision_config.get("strategy_selection_mode", "dynamic")
            fixed_strategy = revision_config.get("fixed_strategy", None)
            strategies = revision_config.get("strategies", None)
            
            self.revision_strategy = AdaptiveRevisionStrategy(
                max_iterations=max_revision_iterations,
                strategies=strategies,
                strategy_selection_mode=strategy_selection_mode,
                fixed_strategy=fixed_strategy
            )
        else:
            self.revision_strategy = None
        
        # Initialize evaluation metrics
        self.evaluator = EvaluationMetrics()
        
        logger.info("Pipeline initialized successfully!")
    
    def generate(
        self,
        query: str,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
        max_revision_iterations: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        num_beams: Optional[int] = None,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate answer with self-verification.
        
        Args:
            query: Query string
            top_k_retrieve: Number of documents to retrieve
            top_k_rerank: Number of documents to rerank
            max_revision_iterations: Maximum revision iterations (overrides init)
            temperature: Sampling temperature (overrides config)
            do_sample: Whether to use sampling (overrides config)
            num_beams: Number of beams for beam search (overrides config)
            **generation_kwargs: Additional generation parameters
        
        Returns:
            Dictionary with generation results and verification
        """
        if max_revision_iterations is None:
            max_revision_iterations = self.max_revision_iterations
        
        # Step 1: Hybrid retrieval
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k_retrieve)
        retrieved_texts = [doc[1] for doc in retrieved_docs]
        retrieved_ids = [doc[0] for doc in retrieved_docs]
        
        # Step 2: Cross-encoder reranking
        # Note: rerank returns (original_index, document, score) tuples
        reranked_docs = self.reranker.rerank(
            query,
            retrieved_texts,
            top_k=top_k_rerank
        )
        # Map back to original IDs using original_index
        reranked_ids = [retrieved_ids[doc[0]] for doc in reranked_docs]
        reranked_texts = [doc[1] for doc in reranked_docs]
        context = " ".join(reranked_texts)
        
        # Step 3: Generation
        # Use generation parameters if provided
        gen_kwargs = {}
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if do_sample is not None:
            gen_kwargs["do_sample"] = do_sample
        if num_beams is not None:
            gen_kwargs["num_beams"] = num_beams
        gen_kwargs.update(generation_kwargs)
        
        generated_text = self.generator.generate(query, context, **gen_kwargs)
        
        # Step 4: Claim extraction
        claims = self.claim_extractor.extract_claims(generated_text)
        
        # Step 5: Verification
        verification_results = self.verifier.verify_generation(
            generated_text,
            reranked_texts,
            claims,
            query=query  # Pass query to help with claim formatting
        )
        
        # Step 6: Adaptive revision (if enabled and verification failed)
        # NOTE: Revision decisions are based ONLY on verification results (entailment checking),
        # NOT on ground truth. This makes the system deployable in production.
        revision_iterations = 0
        revision_history = []  # Track revision history for transparency
        abstained = False
        
        if self.enable_revision and self.revision_strategy:
            if not verification_results.get("verified", False):
                for iteration in range(max_revision_iterations):
                    # Store context before revision for comparison
                    context_before = context
                    retrieved_texts_before = reranked_texts.copy()
                    
                    # Track revision iteration details
                    revision_info = {
                        "iteration": iteration + 1,
                        "generation_before": generated_text,
                        "claims_before": claims,
                        "verification_before": verification_results.copy(),
                        "context_before": context_before[:500] + "..." if len(context_before) > 500 else context_before,  # Truncated for logging
                        "num_contexts_before": len(retrieved_texts_before)
                    }
                    
                    # Use the same top_k_retrieve for revision strategies
                    revised_text, new_verification, strategy_metadata = self.revision_strategy.revise(
                        query=query,
                        initial_generation=generated_text,
                        verification_results=verification_results,
                        retrieval_fn=lambda q, top_k=top_k_retrieve: self.retriever.retrieve(q, top_k=top_k),
                        generation_fn=lambda q, ctx, **kwargs: self.generator.generate(q, ctx, **kwargs),
                        verification_fn=lambda gen, ctxs, clms, q=query: self.verifier.verify_generation(
                            gen, ctxs, clms, query=q
                        ),
                        claim_extractor_fn=lambda text: self.claim_extractor.extract_claims(text),
                        iteration=iteration,
                        top_k_retrieve=top_k_retrieve  # Pass top_k_retrieve to revision strategy
                    )
                    
                    # Extract claims from revised generation
                    revised_claims = self.claim_extractor.extract_claims(revised_text)
                    
                    # Get context after revision from strategy metadata
                    contexts_after = strategy_metadata.get("contexts_used", [])
                    context_after_summary = strategy_metadata.get("context_summary") or strategy_metadata.get("constraint_summary") or strategy_metadata.get("replacement_summary") or "Contexts used in revision"
                    
                    # For re-retrieval, we need to get the actual context text used
                    # The prompt_used contains the context, but we can also reconstruct it
                    if strategy_metadata.get("strategy_name") == "re_retrieval":
                        # Extract context from prompt_used or use contexts_after
                        if contexts_after:
                            context_after_text = " ".join(contexts_after[:3])  # Join first 3 contexts
                        else:
                            # Fallback: extract from prompt if available
                            prompt = strategy_metadata.get("prompt_used", "")
                            if "Context: " in prompt:
                                context_after_text = prompt.split("Context: ")[1].split("...")[0] if "..." in prompt else prompt.split("Context: ")[1]
                            else:
                                context_after_text = "Context retrieved with expanded query"
                    else:
                        # For other strategies, use the contexts from metadata
                        context_after_text = " ".join(contexts_after[:3]) if contexts_after else context_after_summary
                    
                    # Compare verification results to show what changed
                    verification_comparison = self._compare_verification_results(
                        verification_results,
                        new_verification,
                        claims,
                        revised_claims
                    )
                    
                    # Update revision info with strategy metadata
                    revision_info["generation_after"] = revised_text
                    revision_info["claims_after"] = revised_claims
                    revision_info["verification_after"] = new_verification.copy()
                    revision_info["strategy"] = strategy_metadata.get("strategy_name", "unknown")
                    revision_info["prompt_used"] = strategy_metadata.get("prompt_used", None)
                    revision_info["verification_comparison"] = verification_comparison
                    
                    # Add context comparison
                    revision_info["context_after"] = (context_after_text[:500] + "..." if len(context_after_text) > 500 else context_after_text) if isinstance(context_after_text, str) else context_after_summary
                    revision_info["contexts_used"] = contexts_after
                    revision_info["num_contexts_after"] = strategy_metadata.get("num_contexts_retrieved") or strategy_metadata.get("num_contexts", len(contexts_after) if contexts_after else 0)
                    revision_info["context_change_summary"] = f"Contexts: {len(retrieved_texts_before)} → {revision_info['num_contexts_after']} documents"
                    
                    # Add strategy-specific metadata (already includes detailed info from strategy)
                    if strategy_metadata.get("strategy_name") == "re_retrieval":
                        revision_info["expanded_query"] = strategy_metadata.get("expanded_query", None)
                        revision_info["failed_claims_used"] = strategy_metadata.get("failed_claims_used", [])
                        revision_info["context_summary"] = strategy_metadata.get("context_summary", None)
                    elif strategy_metadata.get("strategy_name") == "constrained_generation":
                        revision_info["verified_claims"] = strategy_metadata.get("verified_claims", [])
                        revision_info["constraint_summary"] = strategy_metadata.get("constraint_summary", None)
                    elif strategy_metadata.get("strategy_name") == "claim_by_claim":
                        revision_info["claim_queries"] = strategy_metadata.get("claim_queries", [])
                        revision_info["unverified_claims"] = strategy_metadata.get("unverified_claims", [])
                        revision_info["verified_claims"] = strategy_metadata.get("verified_claims", [])
                        revision_info["claim_replacements"] = strategy_metadata.get("claim_replacements", [])
                        revision_info["replacement_summary"] = strategy_metadata.get("replacement_summary", None)
                    
                    revision_history.append(revision_info)
                    
                    generated_text = revised_text
                    verification_results = new_verification
                    claims = revised_claims  # Update claims to match revised generation
                    revision_iterations += 1
                    
                    if verification_results.get("verified", False):
                        break
                
                # If max iterations reached and still not verified, mark as abstained
                if not verification_results.get("verified", False) and revision_iterations >= max_revision_iterations:
                    abstained = True
                    # Return default "insufficient evidence" response
                    generated_text = "I cannot provide a confident answer based on the available evidence."
                    # Update verification results to reflect abstention
                    verification_results = {
                        "verification_results": verification_results.get("verification_results", []),
                        "num_entailed": verification_results.get("num_entailed", 0),
                        "num_total": verification_results.get("num_total", 0),
                        "entailment_rate": verification_results.get("entailment_rate", 0.0),
                        "avg_entailment_score": verification_results.get("avg_entailment_score", 0.0),
                        "verified": False,
                        "abstained": True
                    }
        
        return {
            "query": query,
            "generated_text": generated_text,
            "retrieved_docs": retrieved_ids,
            "retrieved_texts": retrieved_texts,  # Store for coverage calculation
            "reranked_docs": reranked_ids,  # Use mapped IDs
            "reranked_texts": reranked_texts,  # Store for coverage calculation
            "context": context,
            "claims": claims,
            "verification_results": verification_results,
            "revision_iterations": revision_iterations,
            "revision_history": revision_history,  # Include revision history for transparency
            "verified": verification_results.get("verified", False),
            "abstained": abstained
        }
    
    def evaluate(
        self,
        query: str,
        ground_truth: str,
        relevant_doc_ids: List[int],
        ground_truth_claims: Optional[List[str]] = None,
        **generate_kwargs
    ) -> Dict[str, Any]:
        """
        Generate and evaluate on a query.
        
        Args:
            query: Query string
            ground_truth: Ground truth answer
            relevant_doc_ids: List of relevant document IDs
            ground_truth_claims: Optional list of ground truth claims
            **generate_kwargs: Additional arguments for generate()
        
        Returns:
            Dictionary with generation results and all metrics
        """
        # Generate
        results = self.generate(query, **generate_kwargs)
        
        # Compute metrics
        # Use reranked_texts for coverage (top-k documents used for generation)
        # Get abstention flag from results
        abstained = results.get("abstained", False)
        
        metrics = self.evaluator.compute_all_metrics(
            retrieved_docs=results["retrieved_docs"],
            relevant_docs=relevant_doc_ids,
            verification_results=results["verification_results"]["verification_results"],
            generated=results["generated_text"],
            ground_truth=ground_truth,
            retrieved_texts=results.get("reranked_texts", results.get("retrieved_texts", [])),
            ground_truth_claims=ground_truth_claims,
            abstained=abstained  # Pass abstention flag to exclude from hallucination_rate
        )
        
        results["metrics"] = metrics
        
        return results
    
    def set_entailment_threshold(self, threshold: float):
        """Update entailment threshold."""
        self.verifier.set_threshold(threshold)
    
    def _compare_verification_results(
        self,
        verification_before: Dict[str, Any],
        verification_after: Dict[str, Any],
        claims_before: List[str],
        claims_after: List[str]
    ) -> Dict[str, Any]:
        """
        Compare verification results before and after revision to show what changed.
        
        Returns a dictionary with:
        - entailment_rate_change: Change in entailment rate
        - claim_changes: Per-claim comparison showing score changes
        - summary: Human-readable summary of changes
        """
        before_results = verification_before.get("verification_results", [])
        after_results = verification_after.get("verification_results", [])
        
        entailment_rate_before = verification_before.get("entailment_rate", 0.0)
        entailment_rate_after = verification_after.get("entailment_rate", 0.0)
        entailment_rate_change = entailment_rate_after - entailment_rate_before
        
        # Create a mapping of claims to their verification results
        before_map = {r["claim"]: r for r in before_results}
        after_map = {r["claim"]: r for r in after_results}
        
        # Compare claims
        claim_changes = []
        all_claims = set(list(before_map.keys()) + list(after_map.keys()))
        
        for claim in all_claims:
            before_result = before_map.get(claim)
            after_result = after_map.get(claim)
            
            if before_result and after_result:
                # Same claim, compare scores
                before_score = before_result.get("entailment_score", 0.0)
                after_score = after_result.get("entailment_score", 0.0)
                before_entailed = before_result.get("is_entailed", False)
                after_entailed = after_result.get("is_entailed", False)
                
                claim_changes.append({
                    "claim": claim,
                    "entailment_score_before": before_score,
                    "entailment_score_after": after_score,
                    "score_change": after_score - before_score,
                    "was_entailed_before": before_entailed,
                    "is_entailed_after": after_entailed,
                    "status_change": "improved" if (not before_entailed and after_entailed) else 
                                    "worsened" if (before_entailed and not after_entailed) else
                                    "unchanged" if (before_entailed == after_entailed) else "score_changed"
                })
            elif before_result and not after_result:
                # Claim removed
                claim_changes.append({
                    "claim": claim,
                    "status": "removed",
                    "entailment_score_before": before_result.get("entailment_score", 0.0)
                })
            elif not before_result and after_result:
                # New claim added
                claim_changes.append({
                    "claim": claim,
                    "status": "added",
                    "entailment_score_after": after_result.get("entailment_score", 0.0),
                    "is_entailed_after": after_result.get("is_entailed", False)
                })
        
        # Create summary
        num_improved = sum(1 for c in claim_changes if c.get("status_change") == "improved")
        num_worsened = sum(1 for c in claim_changes if c.get("status_change") == "worsened")
        num_unchanged = sum(1 for c in claim_changes if c.get("status_change") == "unchanged")
        
        summary = (
            f"Entailment rate: {entailment_rate_before:.3f} → {entailment_rate_after:.3f} "
            f"(Δ{entailment_rate_change:+.3f}). "
            f"Claims: {num_improved} improved, {num_worsened} worsened, {num_unchanged} unchanged."
        )
        
        return {
            "entailment_rate_before": entailment_rate_before,
            "entailment_rate_after": entailment_rate_after,
            "entailment_rate_change": entailment_rate_change,
            "num_claims_before": len(claims_before),
            "num_claims_after": len(claims_after),
            "claim_changes": claim_changes,
            "num_improved": num_improved,
            "num_worsened": num_worsened,
            "num_unchanged": num_unchanged,
            "summary": summary
        }

