"""
Entailment-based Factual Verification Module
Uses cross-encoder/nli-deberta-v3-base fine-tuned on NLI tasks
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Any, List, Dict, Tuple, Optional
import numpy as np
import re

from src.utils.device import resolve_device


class EntailmentVerifier:
    """
    Entailment-based verifier for factual claims.
    Uses cross-encoder/nli-deberta-v3-base fine-tuned on NLI tasks.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: str = "auto",
        threshold: float = 0.75,
        max_length: int = 512
    ):
        """
        Initialize entailment verifier.
        
        Args:
            model_name: NLI model name (default: cross-encoder/nli-deberta-v3-base)
            device: Device to run model on
            threshold: Entailment threshold (τ)
            max_length: Maximum sequence length
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        device = resolve_device(device)
        self.device = device
        self.threshold = threshold
        self.max_length = max_length
        
        # Load NLI model for sequence classification
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        self.model.to(device)
        self.model.eval()
        
        # Read the label order from the checkpoint rather than assuming an MNLI
        # convention. NLI checkpoints do not all use the same class indices.
        self.label_map = {
            int(index): str(label).lower()
            for index, label in self.model.config.id2label.items()
        }
        self.contradiction_index = self._label_index("contradiction")
        self.entailment_index = self._label_index("entailment")
        self.neutral_index = self._label_index("neutral")

    def _label_index(self, expected_label: str) -> int:
        """Return the checkpoint index for an NLI label, failing clearly if absent."""
        for index, label in self.label_map.items():
            normalized = label.lower().replace("-", "_").replace(" ", "_")
            if expected_label in normalized:
                return index
        raise ValueError(
            f"Model label mapping {self.label_map!r} does not define "
            f"an '{expected_label}' class. Choose an NLI sequence-classification checkpoint."
        )

    @staticmethod
    def _content_terms(text: str) -> set[str]:
        stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "been", "by", "can",
            "could", "did", "do", "does", "for", "from", "had", "has", "have",
            "her", "his", "how", "in", "is", "it", "its", "of", "on", "or",
            "that", "the", "their", "there", "this", "to", "was", "were",
            "what", "when", "where", "which", "who", "why", "would",
        }
        terms = set()
        for word in re.findall(r"\w+", text.casefold()):
            if word in stop_words or len(word) < 3:
                continue
            if word.endswith("ies") and len(word) > 5:
                word = word[:-3] + "y"
            elif word.endswith("ing") and len(word) > 5:
                word = word[:-3]
                if len(word) > 2 and word[-1] == word[-2]:
                    word = word[:-1]
            elif word.endswith("ed") and len(word) > 4:
                word = word[:-2]
            elif word.endswith("es") and len(word) > 4:
                word = word[:-2]
            elif word.endswith("s") and len(word) > 4 and not word.endswith("ss"):
                word = word[:-1]
            terms.add(word)
        return terms

    @classmethod
    def _answer_sentence_matches_query(cls, claim: str, context: str, query: str) -> bool:
        """Allow a lexical shortcut only when the answer and question share a sentence."""
        question_terms = cls._content_terms(query)
        if not question_terms:
            return False
        answer_tokens = re.findall(r"\w+", claim.casefold())
        if not answer_tokens:
            return False
        fragments = re.split(r"(?<=[.!?])\s+", context)
        sentences = []
        current = ""
        for fragment in fragments:
            current = f"{current} {fragment}".strip()
            if re.search(r"\b(?:Dr|Jr|Mr|Mrs|Ms|Prof|Rep|Sen|Sr|St|U\.S)\.$", current, re.I):
                continue
            sentences.append(current)
            current = ""
        if current:
            sentences.append(current)

        for sentence in sentences:
            sentence_tokens = re.findall(r"\w+", sentence.casefold())
            width = len(answer_tokens)
            if not any(
                sentence_tokens[index:index + width] == answer_tokens
                for index in range(len(sentence_tokens) - width + 1)
            ):
                continue
            if len(question_terms & cls._content_terms(sentence)) >= min(2, len(question_terms)):
                return True
        return False
    
    def verify_claim(
        self,
        claim: str,
        context: str,
        query: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Verify a single claim against context.
        
        Args:
            claim: Claim to verify (hypothesis)
            context: Context to verify against (premise)
            query: Optional query to help format the claim better
        
        Returns:
            Dictionary with 'entailment', 'neutral', 'contradiction' scores
        """
        # Preserve the direct-match shortcut for claims without a question.
        claim_clean = claim.strip()
        claim_normalized_raw = re.sub(r'[^\w\s]', '', claim_clean.lower())
        context_normalized = re.sub(r'[^\w\s]', '', context.lower())
        short_answer_with_query = bool(query and claim_clean and len(claim_clean.split()) <= 10)

        if short_answer_with_query and self._answer_sentence_matches_query(
            claim_clean, context, query
        ):
            return {"contradiction": 0.0, "neutral": 0.0, "entailment": 1.0}
        
        # Check if the raw claim appears as a substring or as a significant phrase
        if not query and claim_normalized_raw and len(claim_normalized_raw.split()) <= 10:
            # For short claims, check if they appear directly in context
            if claim_normalized_raw in context_normalized:
                # For single-word claims, use word boundary matching
                if len(claim_normalized_raw.split()) == 1:
                    # Single word: check word boundaries
                    pattern = r'\b' + re.escape(claim_normalized_raw) + r'\b'
                    if re.search(pattern, context_normalized):
                        return {
                            "contradiction": 0.0,
                            "neutral": 0.0,
                            "entailment": 1.0
                        }
                else:
                    # Multi-word: check if it appears as a phrase
                    # For normalized text, just check substring (already done above)
                    return {
                        "contradiction": 0.0,
                        "neutral": 0.0,
                        "entailment": 1.0
                    }
        
        # Keep the question attached to short answers that need model verification.
        if short_answer_with_query:
            formatted_claim = f"Question: {query.strip()} Answer: {claim_clean.rstrip('.!?')}."
        else:
            formatted_claim = self._format_claim_for_verification(claim, context, query)
        
        # Check if the formatted claim's key content appears in context
        # Extract the actual answer from formatted claim (remove "The answer is" etc.)
        formatted_normalized = re.sub(r'[^\w\s]', '', formatted_claim.lower())
        
        # Check if all significant words from the claim appear in the context
        formatted_words = formatted_normalized.split()
        claim_words = set(formatted_words)
        context_words = set(context_normalized.split())
        
        # Remove common stop words for matching
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'is', 'was', 'are', 'were', 'answer'}
        claim_words_clean = claim_words - stop_words
        context_words_clean = context_words - stop_words
        
        # If most claim words appear in context, check more carefully
        if not query and len(claim_words_clean) > 0:
            overlap_ratio = len(claim_words_clean & context_words_clean) / len(claim_words_clean)
            # For high overlap, check if the key content appears as a phrase
            if overlap_ratio >= 0.8:
                # Check if the core claim (without formatting words) appears as a phrase
                core_claim_words = [
                    word for word in formatted_words
                    if word in claim_words_clean
                ]
                if core_claim_words:
                    # Check if these words appear together in context
                    core_phrase = ' '.join(core_claim_words)
                    if core_phrase in context_normalized:
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
        
        # Get scores
        contradiction_score = float(probs[self.contradiction_index])
        neutral_score = float(probs[self.neutral_index])
        entailment_score = float(probs[self.entailment_index])
        
        # If entailment score is very low but the raw claim appears in context,
        # boost the entailment score (the model might be confused by formatting)
        if not query and entailment_score < 0.3 and claim_normalized_raw in context_normalized:
            # Boost entailment if raw claim is clearly in context
            if len(claim_normalized_raw.split()) <= 5:  # Short answers
                entailment_score = max(entailment_score, 0.7)
                # Adjust other scores proportionally
                other_total = contradiction_score + neutral_score
                if other_total > 0:
                    remaining_probability = 1.0 - entailment_score
                    contradiction_score = contradiction_score * remaining_probability / other_total
                    neutral_score = neutral_score * remaining_probability / other_total
        
        return {
            "contradiction": contradiction_score,
            "neutral": neutral_score,
            "entailment": entailment_score
        }
    
    def _format_claim_for_verification(self, claim: str, context: str, query: Optional[str] = None) -> str:
        """
        Format a claim into a complete sentence for better verification.
        
        For short answers like dates or names, create a more explicit claim
        that can be properly verified against the context.
        """
        claim = claim.strip()
        
        # If claim is already a complete sentence, return as-is
        if claim.endswith('.') or claim.endswith('!') or claim.endswith('?'):
            return claim
        
        # For short claims, try to create a more natural sentence
        if len(claim.split()) <= 5:
            # If we have a query, try to incorporate it for better context
            if query:
                # Try to create a natural statement from query + claim
                # E.g., "Who was Robert's son?" + "Bohemond" -> "Robert's son was Bohemond"
                query_lower = query.lower()
                claim_lower = claim.lower()
                
                # Handle "Who is/was X?" questions
                if query_lower.startswith('who'):
                    # Extract the subject from query if possible
                    # "Who was Robert's son?" -> "Robert's son"
                    # Try to extract the subject after "who is/was"
                    match = re.search(r'who\s+(?:is|was)\s+(.+?)\??$', query_lower)
                    if match:
                        subject = match.group(1).strip()
                        return f"{subject.capitalize()} was {claim}."
                    # Fallback: "X is [claim]"
                    return f"{claim}."
                
                # Handle "What is/was X?" questions
                elif query_lower.startswith('what'):
                    match = re.search(r'what\s+(?:is|was|did|does)\s+(.+?)\??$', query_lower)
                    if match:
                        subject = match.group(1).strip()
                        return f"{subject.capitalize()} is {claim}."
                    return f"{claim}."
                
                # Handle "When" questions
                elif query_lower.startswith('when'):
                    return f"It was {claim}."
                
                # Handle "Where" questions
                elif query_lower.startswith('where'):
                    return f"It was {claim}."
                
                # Default: just return the claim as a statement
                return f"{claim}."
            else:
                # No query available, use simple formatting
                return f"{claim}."
        
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
        threshold: Optional[float] = None,
        query: Optional[str] = None
    ) -> Tuple[bool, float]:
        """
        Check if claim is entailed by context (above threshold).
        
        Args:
            claim: Claim to verify
            context: Context to verify against
            threshold: Entailment threshold (default: self.threshold)
            query: Optional query to help format the claim better
        
        Returns:
            Tuple of (is_entailed: bool, entailment_score: float)
        """
        if threshold is None:
            threshold = self.threshold
        
        result = self.verify_claim(claim, context, query)
        entailment_score = result["entailment"]
        is_entailed = entailment_score >= threshold
        
        return is_entailed, entailment_score
    
    def verify_generation(
        self,
        generated_text: str,
        retrieved_contexts: List[str],
        claims: List[str],
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify all claims in generated text against retrieved contexts.
        
        Args:
            generated_text: Generated text
            retrieved_contexts: List of retrieved context documents
            claims: List of extracted claims
            query: Optional query to help format claims better
        
        Returns:
            Dictionary with verification results
        """
        # Combine contexts for verification
        combined_context = " ".join(retrieved_contexts[:3]) if retrieved_contexts else ""  # Use top 3 contexts
        
        verification_results = []
        for claim in claims:
            # Get full verification result (contradiction, neutral, entailment scores)
            full_result = self.verify_claim(claim, combined_context, query)

            # Determine label based on scores
            entailment_score = full_result["entailment"]
            contradiction_score = full_result["contradiction"]
            neutral_score = full_result["neutral"]
            is_entailed = entailment_score >= self.threshold
            
            # Label: highest probability wins, but be more careful about contradiction
            if entailment_score >= self.threshold:
                label = "ENTAILED"
            elif contradiction_score > neutral_score and contradiction_score > 0.6:
                # Only mark as contradicted if contradiction score is high
                label = "CONTRADICTED"
            elif entailment_score > 0.3 and entailment_score > contradiction_score:
                # If entailment is moderate and higher than contradiction, mark as NO_EVIDENCE
                # (might be entailed but below threshold)
                label = "NO_EVIDENCE"
            else:
                label = "NO_EVIDENCE"
            
            verification_results.append({
                "claim": claim,
                "is_entailed": is_entailed,
                "entailment_score": entailment_score,
                "contradiction_score": contradiction_score,
                "neutral_score": neutral_score,
                "label": label,
                "threshold": self.threshold
            })
        
        # Compute aggregate metrics
        num_entailed = sum(1 for r in verification_results if r["is_entailed"])
        num_total = len(verification_results)
        entailment_rate = num_entailed / num_total if num_total > 0 else 0.0
        # Handle empty list to avoid NaN
        if verification_results:
            avg_score = float(np.mean([r["entailment_score"] for r in verification_results]))
        else:
            avg_score = 0.0
        
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
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold
