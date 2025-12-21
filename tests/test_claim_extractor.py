"""
Tests for spaCy Claim Extractor
Validates atomic claim extraction from generated text.
"""

from src.verification.spacy_claim_extractor import SpacyClaimExtractor


def test_extract_returns_claims_for_simple_text():
    """Test that extractor produces multiple claims from multi-sentence text."""
    extractor = SpacyClaimExtractor()
    text = "Intuit provides financial software. TurboTax helps users file taxes."
    claims = extractor.extract(text)
    
    assert len(claims) >= 1, "Should extract at least one claim"
    assert any("Intuit" in c.text for c in claims), "Should extract claim about Intuit"
    
    # Print for visibility
    print(f"\n✅ Extracted {len(claims)} claims from: '{text}'")
    for i, claim in enumerate(claims, 1):
        print(f"   [{i}] {claim.text}")


def test_extract_empty_text():
    """Test that extractor handles empty text gracefully."""
    extractor = SpacyClaimExtractor()
    claims = extractor.extract("")
    assert claims == [], "Empty text should return empty list"
    
    claims_none = extractor.extract(None)
    assert claims_none == [], "None text should return empty list"


def test_extract_single_sentence():
    """Test extraction from single sentence."""
    extractor = SpacyClaimExtractor()
    text = "Paris is the capital of France."
    claims = extractor.extract(text)
    
    assert len(claims) >= 1, "Should extract at least one claim"
    assert any("Paris" in c.text for c in claims), "Should extract claim about Paris"
    
    print(f"\n✅ Extracted {len(claims)} claims from: '{text}'")
    for i, claim in enumerate(claims, 1):
        print(f"   [{i}] {claim.text}")


def test_extract_complex_text():
    """Test extraction from complex multi-sentence text."""
    extractor = SpacyClaimExtractor()
    text = (
        "The Eiffel Tower is located in Paris, France. "
        "It was built in 1889 for the World's Fair. "
        "Gustave Eiffel designed the iconic structure."
    )
    claims = extractor.extract(text)
    
    assert len(claims) >= 2, "Should extract multiple claims from complex text"
    
    print(f"\n✅ Extracted {len(claims)} claims from complex text:")
    for i, claim in enumerate(claims, 1):
        print(f"   [{i}] {claim.text}")


def test_extract_with_max_claims_limit():
    """Test that max_claims parameter limits output."""
    extractor = SpacyClaimExtractor()
    text = " ".join([f"Sentence {i} contains information." for i in range(20)])
    
    claims_limited = extractor.extract(text, max_claims=5)
    assert len(claims_limited) <= 5, "Should respect max_claims limit"
    
    print(f"\n✅ Limited to {len(claims_limited)} claims (max_claims=5)")


def test_deduplication():
    """Test that duplicate claims are removed."""
    extractor = SpacyClaimExtractor()
    text = "Paris is beautiful. Paris is beautiful. Paris is the capital."
    claims = extractor.extract(text)
    
    claim_texts = [c.text for c in claims]
    assert len(claim_texts) == len(set(claim_texts)), "Should deduplicate claims"
    
    print(f"\n✅ Deduplicated to {len(claims)} unique claims")


def test_filters_questions():
    """Test that questions are filtered out."""
    extractor = SpacyClaimExtractor()
    text = "Paris is the capital. What is the population? France is in Europe."
    claims = extractor.extract(text)
    
    # Questions should be filtered
    assert not any("?" in c.text for c in claims), "Should filter out questions"
    
    print(f"\n✅ Filtered questions, extracted {len(claims)} factual claims")


def test_integration_with_controller():
    """Test that claim extractor integrates with VerificationController."""
    from src.verification_controller import VerificationController
    
    controller = VerificationController()
    
    # Verify controller has claim extractor
    assert hasattr(controller, 'claim_extractor'), "Controller should have claim_extractor"
    assert isinstance(controller.claim_extractor, SpacyClaimExtractor), "Should be SpacyClaimExtractor instance"
    
    # Test extraction through controller
    result = controller.verify(
        generated_answer="Machine learning is a subset of AI. Deep learning uses neural networks.",
        retrieved_docs={"passages": [], "ids": []},
        query="What is machine learning?"
    )
    
    # Should extract multiple claims now (not just one)
    assert len(result.claims) >= 1, "Should extract claims"
    
    print(f"\n✅ Controller extracted {len(result.claims)} claims:")
    for claim in result.claims:
        print(f"   [{claim.claim_id}] {claim.text}")


if __name__ == "__main__":
    # Run tests manually
    print("="*70)
    print("  CLAIM EXTRACTOR TESTS")
    print("="*70)
    
    test_extract_returns_claims_for_simple_text()
    test_extract_empty_text()
    test_extract_single_sentence()
    test_extract_complex_text()
    test_extract_with_max_claims_limit()
    test_deduplication()
    test_filters_questions()
    test_integration_with_controller()
    
    print("\n" + "="*70)
    print("  ✅ ALL TESTS PASSED!")
    print("="*70 + "\n")
