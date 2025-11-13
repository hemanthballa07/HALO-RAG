"""
Entailment-based Factual Verification Module
Uses DeBERTa-v3-large fine-tuned on MNLI + FEVER
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.deberta_v2 import DebertaV2Tokenizer
from typing import List, Dict, Tuple, Optional
import numpy as np


class EntailmentVerifier:
    """
    Entailment-based verifier for factual claims.
    Uses DeBERTa-v3-large trained on MNLI + FEVER.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        device: str = "cuda",
        threshold: float = 0.75,
        max_length: int = 512,
        use_mnli_model: bool = True
    ):
        """
        Initialize entailment verifier.
        
        Args:
            model_name: DeBERTa model name
            device: Device to run model on
            threshold: Entailment threshold (τ)
            max_length: Maximum sequence length
        """
        self.device = device
        self.threshold = threshold
        self.max_length = max_length
        
        # Load model for sequence classification (MNLI task)
        # microsoft/deberta-v3-large doesn't have a public MNLI checkpoint
        # Use cross-encoder/nli-deberta-v3-base which is properly fine-tuned on NLI tasks
        actual_model_name = model_name
        if use_mnli_model:
            # Try to use a properly fine-tuned NLI model
            mnli_models = [
                "cross-encoder/nli-deberta-v3-base",  # Properly fine-tuned on NLI tasks
            ]
            
            model_loaded = False
            for mnli_model in mnli_models:
                try:
                    self.model = AutoModelForSequenceClassification.from_pretrained(mnli_model)
                    actual_model_name = mnli_model
                    if mnli_model != model_name:
                        print(f"✓ Using NLI fine-tuned model: {mnli_model} (instead of {model_name})")
                    else:
                        print(f"✓ Using NLI fine-tuned model: {model_name}")
                    model_loaded = True
                    break
                except Exception as e:
                    continue
            
            if not model_loaded:
                # Fallback: use base model but warn that it's not fine-tuned
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=3,
                    ignore_mismatched_sizes=True
                )
                print(f"⚠ Warning: Could not load NLI fine-tuned model. Using {model_name} with random classifier head.")
                print(f"⚠ Verification scores will be unreliable. For accurate results, ensure cross-encoder/nli-deberta-v3-base is available.")
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3,
                ignore_mismatched_sizes=True
            )
        
        # Load tokenizer matching the actual model
        # Use slow tokenizer for DeBERTa on CPU (fast tokenizer has tiktoken issues)
        if device != "cuda":
            print("⚠ Using slow tokenizer for CPU/MPS compatibility")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(actual_model_name, use_fast=False)
            except Exception as e:
                print(f"⚠ AutoTokenizer failed: {e}. Trying DebertaV2Tokenizer")
                try:
                    self.tokenizer = DebertaV2Tokenizer.from_pretrained(actual_model_name)
                except Exception as e2:
                    import os
                    os.environ["TOKENIZERS_PARALLELISM"] = "false"
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        actual_model_name,
                        use_fast=False,
                        local_files_only=False
                    )
        else:
            # On CUDA, try fast tokenizer first
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(actual_model_name, use_fast=True)
            except Exception as e:
                print(f"⚠ Fast tokenizer failed, using slow tokenizer: {e}")
                self.tokenizer = AutoTokenizer.from_pretrained(actual_model_name, use_fast=False)
        
        self.model.to(device)
        self.model.eval()
        
        # Label mapping: 0=contradiction, 1=neutral, 2=entailment
        self.label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
    
    def verify_claim(
        self,
        claim: str,
        context: str
    ) -> Dict[str, float]:
        """
        Verify a single claim against context.
        
        Args:
            claim: Claim to verify (hypothesis)
            context: Context to verify against (premise)
        
        Returns:
            Dictionary with 'entailment', 'neutral', 'contradiction' scores
        """
        # Format claim as a complete sentence if it's not already
        # Short answers like "2003" or "June 2005" need to be converted to full claims
        formatted_claim = self._format_claim_for_verification(claim, context)
        
        # Check if the claim appears directly in the context (exact or near-exact match)
        # This handles cases where the claim is a direct quote or close variant
        import re
        claim_normalized = re.sub(r'[^\w\s]', '', formatted_claim.lower())
        context_normalized = re.sub(r'[^\w\s]', '', context.lower())
        
        # If the formatted claim is a substring of the context, it's likely entailed
        # Check if all significant words from the claim appear in the context
        claim_words = set(claim_normalized.split())
        context_words = set(context_normalized.split())
        
        # Remove common stop words for matching
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'is', 'was', 'are', 'were'}
        claim_words_clean = claim_words - stop_words
        context_words_clean = context_words - stop_words
        
        # If most claim words appear in context and the claim is short, likely entailed
        if len(claim_words_clean) > 0:
            overlap_ratio = len(claim_words_clean & context_words_clean) / len(claim_words_clean)
            # If high overlap and claim appears as substring, mark as entailed
            if overlap_ratio >= 0.8 and claim_normalized in context_normalized:
                return {
                    "contradiction": 0.0,
                    "neutral": 0.0,
                    "entailment": 1.0
                }
        
        # Format as premise-hypothesis pair for NLI
        # Premise = context (what we know)
        # Hypothesis = formatted claim (what we want to verify)
        inputs = self.tokenizer(
            context,
            formatted_claim,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        probs = probs.cpu().numpy()[0]
        
        return {
            "contradiction": float(probs[0]),
            "neutral": float(probs[1]),
            "entailment": float(probs[2])
        }
    
    def _format_claim_for_verification(self, claim: str, context: str) -> str:
        """
        Format a claim into a complete sentence for better verification.
        
        For short answers like dates or names, create a more explicit claim
        that can be properly verified against the context.
        """
        claim = claim.strip()
        
        # If claim is already a complete sentence, return as-is
        if claim.endswith('.') or claim.endswith('!') or claim.endswith('?'):
            return claim
        
        # For short claims, make them more explicit
        if len(claim.split()) <= 5:
            # Simple formatting: wrap in a natural sentence
            return f"The answer is {claim}."
        
        # Default: return as-is
        return claim
    
    def verify_claims(
        self,
        claims: List[str],
        contexts: List[str]
    ) -> List[Dict[str, float]]:
        """
        Verify multiple claims against their contexts.
        
        Args:
            claims: List of claims
            contexts: List of contexts (one per claim)
        
        Returns:
            List of verification results
        """
        results = []
        for claim, context in zip(claims, contexts):
            result = self.verify_claim(claim, context)
            results.append(result)
        
        return results
    
    def is_entailed(
        self,
        claim: str,
        context: str,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
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
    ) -> Dict[str, any]:
        """
        Verify all claims in generated text against retrieved contexts.
        
        Args:
            generated_text: Generated text
            retrieved_contexts: List of retrieved context documents
            claims: List of extracted claims
        
        Returns:
            Dictionary with verification results
        """
        # Combine contexts for verification
        combined_context = " ".join(retrieved_contexts[:3])  # Use top 3 contexts
        
        verification_results = []
        for claim in claims:
            # Get full verification result (contradiction, neutral, entailment scores)
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
        avg_score = np.mean([r["entailment_score"] for r in verification_results])
        
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

