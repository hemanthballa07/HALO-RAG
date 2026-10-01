"""
Claim Extraction Module using spaCy for Subject-Verb-Object (SVO) extraction.
"""

import spacy
from typing import List, Dict


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
        return [triple for sent in doc.sents for triple in self._extract_svo_from_sentence(sent)]

    def _extract_svo_from_sentence(self, sent) -> List[Dict[str, str]]:
        triples = []
        for token in sent:
            if token.pos_ != "VERB" or token.dep_ != "ROOT":
                continue

            subject = None
            obj = None
            verb_tokens = [token]
            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"] and subject is None:
                    subject = self._get_phrase(child)
                elif child.dep_ in ["dobj", "pobj", "attr"] and obj is None:
                    obj = self._get_phrase(child)
                elif child.dep_ in ["aux", "auxpass", "neg"]:
                    verb_tokens.append(child)

            if subject and obj:
                verb = " ".join(part.text for part in sorted(verb_tokens, key=lambda part: part.i))
                triples.append({
                    "subject": subject,
                    "verb": verb,
                    "object": obj,
                    "claim": f"{subject} {verb} {obj}"
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
        if not text.strip():
            return []

        claims = []
        seen = set()
        for sent in self.nlp(text).sents:
            sentence = sent.text.strip()
            if not sentence or sentence.endswith("?"):
                continue

            triples = self._extract_svo_from_sentence(sent)
            sentence_claims = [triple["claim"] for triple in triples] or [sentence]
            for claim in sentence_claims:
                if claim not in seen:
                    seen.add(claim)
                    claims.append(claim)

        return claims
    
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
