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
        max_revision_iterations: int = 3
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
            self.revision_strategy = AdaptiveRevisionStrategy(
                max_iterations=max_revision_iterations
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
            claims
        )
        
        # Step 6: Adaptive revision (if enabled and verification failed)
        revision_iterations = 0
        revision_history = []  # Track revision history for transparency
        abstained = False
        
        if self.enable_revision and self.revision_strategy:
            if not verification_results.get("verified", False):
                for iteration in range(max_revision_iterations):
                    # Track revision iteration details
                    revision_info = {
                        "iteration": iteration + 1,
                        "generation_before": generated_text,
                        "claims_before": claims,
                        "verification_before": verification_results.copy()
                    }
                    
                    revised_text, new_verification, strategy_metadata = self.revision_strategy.revise(
                        query=query,
                        initial_generation=generated_text,
                        verification_results=verification_results,
                        retrieval_fn=lambda q, top_k=20: self.retriever.retrieve(q, top_k=top_k),
                        generation_fn=lambda q, ctx, **kwargs: self.generator.generate(q, ctx, **kwargs),
                        verification_fn=lambda gen, ctxs, clms: self.verifier.verify_generation(
                            gen, ctxs, clms
                        ),
                        claim_extractor_fn=lambda text: self.claim_extractor.extract_claims(text),
                        iteration=iteration
                    )
                    
                    # Extract claims from revised generation
                    revised_claims = self.claim_extractor.extract_claims(revised_text)
                    
                    # Update revision info with strategy metadata
                    revision_info["generation_after"] = revised_text
                    revision_info["claims_after"] = revised_claims
                    revision_info["verification_after"] = new_verification.copy()
                    revision_info["strategy"] = strategy_metadata.get("strategy_name", "unknown")
                    revision_info["prompt_used"] = strategy_metadata.get("prompt_used", None)
                    
                    # Add strategy-specific metadata
                    if strategy_metadata.get("strategy_name") == "re_retrieval":
                        revision_info["expanded_query"] = strategy_metadata.get("expanded_query", None)
                        revision_info["failed_claims_used"] = strategy_metadata.get("failed_claims_used", [])
                    elif strategy_metadata.get("strategy_name") == "constrained_generation":
                        revision_info["verified_claims"] = strategy_metadata.get("verified_claims", [])
                    elif strategy_metadata.get("strategy_name") == "claim_by_claim":
                        revision_info["claim_queries"] = strategy_metadata.get("claim_queries", [])
                        revision_info["unverified_claims"] = strategy_metadata.get("unverified_claims", [])
                        revision_info["verified_claims"] = strategy_metadata.get("verified_claims", [])
                    
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

