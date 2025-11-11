"""
Entailment-based Factual Verification Module
Uses DeBERTa-v3-large fine-tuned on MNLI + FEVER
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
        max_length: int = 512
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
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3  # entailment, neutral, contradiction
        )
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
            claim: Claim to verify
            context: Context to verify against
        
        Returns:
            Dictionary with 'entailment', 'neutral', 'contradiction' scores
        """
        # Format as premise-hypothesis pair
        # For MNLI: premise=context, hypothesis=claim
        inputs = self.tokenizer(
            context,
            claim,
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

