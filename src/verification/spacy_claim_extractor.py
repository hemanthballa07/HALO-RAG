"""
spaCy-based Claim Extractor
Extracts atomic claims from generated text using dependency parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import spacy


@dataclass
class ExtractedClaim:
    """Represents an extracted claim before verification."""
    text: str


class SpacyClaimExtractor:
    """
    Extracts atomic claims from generated text using spaCy.
    
    Strategy:
      1) Sentence segmentation
      2) For each sentence, extract SVO-style claims where possible
      3) Fallback: keep the sentence as a claim (if it's short and declarative)
    """

    def __init__(self, model: str = "en_core_web_sm"):
        """
        Initialize claim extractor with spaCy model.
        
        Args:
            model: spaCy model name (default: en_core_web_sm)
        """
        self.nlp = spacy.load(model)

    def extract(self, text: str, max_claims: int = 12) -> List[ExtractedClaim]:
        """
        Extract atomic claims from text.
        
        Args:
            text: Generated text to extract claims from
            max_claims: Maximum number of claims to extract
        
        Returns:
            List of ExtractedClaim objects
        """
        text = (text or "").strip()
        if not text:
            return []

        doc = self.nlp(text)
        claims: List[ExtractedClaim] = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            # Try to extract one or more structured claims from the sentence
            structured = self._extract_svo_from_sentence(sent)
            if structured:
                claims.extend(structured)
            else:
                # Fallback: treat sentence as a claim if it looks factual
                if self._looks_like_factual_statement(sent):
                    claims.append(ExtractedClaim(text=self._normalize(sent_text)))

            if len(claims) >= max_claims:
                break

        # Deduplicate while preserving order
        seen = set()
        deduped: List[ExtractedClaim] = []
        for c in claims:
            if c.text not in seen:
                seen.add(c.text)
                deduped.append(c)

        return deduped[:max_claims]

    def _extract_svo_from_sentence(self, sent) -> List[ExtractedClaim]:
        """
        Extract minimal SVO-ish propositions from one sentence.
        
        spaCy dependency parse heuristic:
          - subject: nsubj / nsubjpass
          - verb: ROOT (or main verb)
          - object/complement: dobj / pobj / attr / acomp / dative / oprd
        
        Args:
            sent: spaCy Span representing a sentence
        
        Returns:
            List of ExtractedClaim objects
        """
        root = next((t for t in sent if t.dep_ == "ROOT"), None)
        if root is None:
            return []

        # Find subject(s)
        subjects = [t for t in root.lefts if t.dep_ in ("nsubj", "nsubjpass", "csubj")]
        if not subjects:
            # Sometimes subject isn't direct left child; search within sentence
            subjects = [t for t in sent if t.dep_ in ("nsubj", "nsubjpass", "csubj")]

        # Find object/complement candidates
        objs = [t for t in root.rights if t.dep_ in ("dobj", "pobj", "attr", "acomp", "dative", "oprd")]
        if not objs:
            objs = [t for t in sent if t.dep_ in ("dobj", "pobj", "attr", "acomp", "dative", "oprd")]

        # If we can't form at least Subject + Verb, skip structured extraction
        if not subjects:
            return []

        verb_phrase = self._verb_phrase(root)

        claims: List[ExtractedClaim] = []
        for subj in subjects[:2]:  # cap to avoid explosion
            subj_span = self._expand_np(subj)
            subj_text = self._normalize(subj_span.text)

            if objs:
                for obj in objs[:3]:
                    obj_span = self._expand_np(obj)
                    obj_text = self._normalize(obj_span.text)

                    # Build claim: "<subj> <verb> <obj>"
                    claim = f"{subj_text} {verb_phrase} {obj_text}".strip()
                    if self._valid_claim_text(claim):
                        claims.append(ExtractedClaim(text=claim))
            else:
                # No object, still form: "<subj> <verb_phrase>"
                claim = f"{subj_text} {verb_phrase}".strip()
                if self._valid_claim_text(claim):
                    claims.append(ExtractedClaim(text=claim))

        return claims

    def _expand_np(self, token):
        """
        Expand token to its noun chunk if present; otherwise expand to subtree.
        
        Args:
            token: spaCy Token
        
        Returns:
            spaCy Span representing the expanded noun phrase
        """
        # Prefer noun_chunks when token is inside one
        doc = token.doc
        for chunk in doc.noun_chunks:
            if token.i >= chunk.start and token.i < chunk.end:
                return chunk
        # fallback: subtree span
        subtree_tokens = list(token.subtree)
        return doc[subtree_tokens[0].i : subtree_tokens[-1].i + 1]

    def _verb_phrase(self, root) -> str:
        """
        Build a lightweight verb phrase including auxiliaries/negation.
        
        Args:
            root: spaCy Token representing the root verb
        
        Returns:
            Normalized verb phrase string
        """
        parts = []
        # include aux verbs and negation left of root
        for t in root.lefts:
            if t.dep_ in ("aux", "auxpass", "neg"):
                parts.append(t.text)
        parts.append(root.lemma_)  # use lemma for normalization
        return self._normalize(" ".join(parts))

    def _looks_like_factual_statement(self, sent) -> bool:
        """
        Heuristic: avoid questions and very short/fragmented lines.
        
        Args:
            sent: spaCy Span representing a sentence
        
        Returns:
            True if sentence looks like a factual statement
        """
        text = sent.text.strip()
        if text.endswith("?"):
            return False
        if len(text) < 12:
            return False
        return True

    def _valid_claim_text(self, claim: str) -> bool:
        """
        Validate that claim text is substantive enough.
        
        Args:
            claim: Claim text to validate
        
        Returns:
            True if claim is valid
        """
        claim = claim.strip()
        if len(claim) < 8:
            return False
        if claim.endswith("?"):
            return False
        return True

    def _normalize(self, s: str) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            s: Text to normalize
        
        Returns:
            Normalized text
        """
        return " ".join(s.split())
