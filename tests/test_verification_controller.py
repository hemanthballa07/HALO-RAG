"""
Test suite for Verification Controller
Validates the verification flow without ML models.
"""

import pytest
from src.verification_controller import VerificationController, Claim, VerificationResult


def test_controller_initialization():
    """Test that controller initializes with correct defaults."""
    controller = VerificationController()
    
    assert controller.entailment_threshold == 0.75
    assert controller.max_revision_cycles == 2
    assert controller.enable_revision is True


def test_controller_custom_config():
    """Test controller with custom configuration."""
    controller = VerificationController(
        entailment_threshold=0.85,
        max_revision_cycles=3,
        enable_revision=False
    )
    
    assert controller.entailment_threshold == 0.85
    assert controller.max_revision_cycles == 3
    assert controller.enable_revision is False


def test_verify_returns_structured_result():
    """Test that verify() returns a properly structured VerificationResult."""
    controller = VerificationController()
    
    generated_answer = "The Eiffel Tower is located in Paris, France."
    retrieved_docs = {
        "passages": ["Paris is the capital of France.", "The Eiffel Tower was built in 1889."],
        "ids": ["P1", "P2"]
    }
    
    result = controller.verify(
        generated_answer=generated_answer,
        retrieved_docs=retrieved_docs,
        query="Where is the Eiffel Tower?"
    )
    
    # Validate result structure
    assert isinstance(result, VerificationResult)
    assert isinstance(result.claims, list)
    assert result.revision_cycles == 0
    assert isinstance(result.reason, str)
    
    # Validate claims
    assert len(result.claims) > 0
    for claim in result.claims:
        assert isinstance(claim, Claim)
        assert hasattr(claim, 'claim_id')
        assert hasattr(claim, 'text')
        assert hasattr(claim, 'entailment')
        assert hasattr(claim, 'confidence')


def test_verify_empty_answer():
    """Test verification with empty answer."""
    controller = VerificationController()
    
    result = controller.verify(
        generated_answer="",
        retrieved_docs={"passages": [], "ids": []}
    )
    
    assert isinstance(result, VerificationResult)
    assert len(result.claims) == 0
    assert result.total_claims == 0


def test_claim_extraction_stub():
    """Test that claim extraction creates valid Claim objects."""
    controller = VerificationController()
    
    text = "Machine learning is a subset of artificial intelligence."
    claims = controller._extract_claims(text)
    
    assert len(claims) == 1  # Stub returns single claim
    assert claims[0].claim_id == "C1"
    assert claims[0].text == text
    assert claims[0].entailment == "UNVERIFIED"
    assert claims[0].confidence == 0.0


def test_verification_result_schema_validation():
    """Test that VerificationResult validates correctly."""
    # Valid result
    result = VerificationResult(
        claims=[
            Claim(claim_id="C1", text="Test claim", entailment="ENTAILMENT", confidence=0.9)
        ],
        verified=True,
        revision_cycles=0,
        reason="All claims verified",
        total_claims=1,
        verified_claims=1,
        hallucinated_claims=0,
        avg_confidence=0.9
    )
    
    assert result.verified is True
    assert result.total_claims == 1
    assert result.verified_claims == 1


def test_claim_schema_validation():
    """Test that Claim schema validates correctly."""
    # Valid claim
    claim = Claim(
        claim_id="C1",
        text="Test claim",
        evidence_ids=["P1", "P2"],
        entailment="ENTAILMENT",
        confidence=0.85
    )
    
    assert claim.claim_id == "C1"
    assert claim.confidence == 0.85
    assert len(claim.evidence_ids) == 2
    
    # Test confidence bounds
    with pytest.raises(Exception):  # Pydantic validation error
        Claim(claim_id="C2", text="Invalid", confidence=1.5)  # > 1.0


if __name__ == "__main__":
    # Run tests manually
    test_controller_initialization()
    test_controller_custom_config()
    test_verify_returns_structured_result()
    test_verify_empty_answer()
    test_claim_extraction_stub()
    test_verification_result_schema_validation()
    test_claim_schema_validation()
    
    print("✅ All tests passed!")
