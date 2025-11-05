"""
Claim Extraction Module using spaCy for Subject-Verb-Object (SVO) extraction.
"""

import spacy
from typing import List, Dict, Tuple
import re


class ClaimExtractor:
    """
    Extract factual claims from generated text using spaCy SVO extraction.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize claim extractor.
        
        Args:
            model_name: spaCy model name
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise ValueError(
                f"spaCy model '{model_name}' not found. "
                f"Install with: python -m spacy download {model_name}"
            )
    
    def extract_svo_triples(self, text: str) -> List[Dict[str, str]]:
        """
        Extract Subject-Verb-Object triples from text.
        
        Args:
            text: Input text to extract claims from
        
        Returns:
            List of dictionaries with 'subject', 'verb', 'object' keys
        """
        doc = self.nlp(text)
        triples = []
        
        for sent in doc.sents:
            # Extract SVO triples from each sentence
            for token in sent:
                if token.pos_ == "VERB" and token.dep_ == "ROOT":
                    # Find subject
                    subject = None
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subject = self._get_phrase(child)
                            break
                    
                    # Find object
                    obj = None
                    for child in token.children:
                        if child.dep_ in ["dobj", "pobj", "attr"]:
                            obj = self._get_phrase(child)
                            break
                    
                    if subject and obj:
                        triples.append({
                            "subject": subject,
                            "verb": token.text,
                            "object": obj,
                            "claim": f"{subject} {token.text} {obj}"
                        })
        
        return triples
    
    def _get_phrase(self, token) -> str:
        """Get complete phrase for a token including its modifiers."""
        phrase_tokens = [token]
        
        # Add modifiers
        for child in token.children:
            if child.dep_ in ["det", "amod", "compound", "prep"]:
                phrase_tokens.append(child)
        
        # Sort by position in sentence
        phrase_tokens.sort(key=lambda t: t.i)
        
        return " ".join([t.text for t in phrase_tokens])
    
    def extract_claims(self, text: str) -> List[str]:
        """
        Extract claims as strings from text.
        
        Args:
            text: Input text
        
        Returns:
            List of claim strings
        """
        triples = self.extract_svo_triples(text)
        return [t["claim"] for t in triples]
    
    def extract_claims_with_context(
        self,
        text: str,
        context: str = None
    ) -> List[Dict[str, str]]:
        """
        Extract claims with their context.
        
        Args:
            text: Generated text
            context: Retrieved context (optional)
        
        Returns:
            List of dictionaries with 'claim', 'context' keys
        """
        triples = self.extract_svo_triples(text)
        
        claims_with_context = []
        for triple in triples:
            claims_with_context.append({
                "claim": triple["claim"],
                "subject": triple["subject"],
                "verb": triple["verb"],
                "object": triple["object"],
                "context": context
            })
        
        return claims_with_context

