#!/usr/bin/env python3
"""
Verification Controller Demo
Demonstrates the verification flow without requiring ML models.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.verification_controller import VerificationController, VerificationResult


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def demo_basic_verification():
    """Demonstrate basic verification flow."""
    print_section("Demo 1: Basic Verification")
    
    # Initialize controller
    controller = VerificationController(
        entailment_threshold=0.75,
        max_revision_cycles=2,
        enable_revision=True
    )
    
    print(f"✅ Controller initialized:")
    print(f"   - Entailment threshold: {controller.entailment_threshold}")
    print(f"   - Max revision cycles: {controller.max_revision_cycles}")
    print(f"   - Revision enabled: {controller.enable_revision}")
    
    # Simulate a generated answer
    generated_answer = "The Eiffel Tower is located in Paris, France and was completed in 1889."
    
    # Simulate retrieved documents
    retrieved_docs = {
        "passages": [
            "Paris is the capital and largest city of France.",
            "The Eiffel Tower was built between 1887 and 1889.",
            "Gustave Eiffel designed the tower for the 1889 World's Fair."
        ],
        "ids": ["P1", "P2", "P3"]
    }
    
    print(f"\n📝 Generated Answer:")
    print(f"   {generated_answer}")
    
    print(f"\n📚 Retrieved Evidence ({len(retrieved_docs['passages'])} passages):")
    for i, passage in enumerate(retrieved_docs['passages'], 1):
        print(f"   {i}. {passage}")
    
    # Run verification
    print(f"\n🔍 Running verification...")
    result = controller.verify(
        generated_answer=generated_answer,
        retrieved_docs=retrieved_docs,
        query="Where is the Eiffel Tower and when was it built?"
    )
    
    # Display results
    print(f"\n📊 Verification Results:")
    print(f"   - Verified: {result.verified}")
    print(f"   - Total claims: {result.total_claims}")
    print(f"   - Verified claims: {result.verified_claims}")
    print(f"   - Hallucinated claims: {result.hallucinated_claims}")
    print(f"   - Revision cycles: {result.revision_cycles}")
    print(f"   - Reason: {result.reason}")
    
    print(f"\n🔬 Extracted Claims:")
    for claim in result.claims:
        print(f"   - [{claim.claim_id}] {claim.text}")
        print(f"     Entailment: {claim.entailment} (confidence: {claim.confidence:.2f})")


def demo_empty_answer():
    """Demonstrate handling of empty answers."""
    print_section("Demo 2: Empty Answer Handling")
    
    controller = VerificationController()
    
    result = controller.verify(
        generated_answer="",
        retrieved_docs={"passages": [], "ids": []}
    )
    
    print(f"📝 Generated Answer: (empty)")
    print(f"\n📊 Verification Results:")
    print(f"   - Verified: {result.verified}")
    print(f"   - Total claims: {result.total_claims}")
    print(f"   - Reason: {result.reason}")


def demo_custom_configuration():
    """Demonstrate custom controller configuration."""
    print_section("Demo 3: Custom Configuration")
    
    # High-precision configuration
    high_precision = VerificationController(
        entailment_threshold=0.90,  # Very strict
        max_revision_cycles=3,
        enable_revision=True
    )
    
    print("🎯 High-Precision Configuration:")
    print(f"   - Threshold: {high_precision.entailment_threshold} (strict)")
    print(f"   - Max cycles: {high_precision.max_revision_cycles}")
    
    # Fast configuration (no revision)
    fast_mode = VerificationController(
        entailment_threshold=0.70,  # More lenient
        max_revision_cycles=0,
        enable_revision=False
    )
    
    print("\n⚡ Fast Mode Configuration:")
    print(f"   - Threshold: {fast_mode.entailment_threshold} (lenient)")
    print(f"   - Revision: {fast_mode.enable_revision} (disabled for speed)")


def demo_schema_validation():
    """Demonstrate Pydantic schema validation."""
    print_section("Demo 4: Schema Validation")
    
    from src.verification_controller import Claim
    
    # Valid claim
    print("✅ Creating valid claim:")
    claim = Claim(
        claim_id="C1",
        text="Machine learning is a subset of AI.",
        evidence_ids=["P1", "P2"],
        entailment="ENTAILMENT",
        confidence=0.92
    )
    print(f"   {claim.claim_id}: {claim.text}")
    print(f"   Confidence: {claim.confidence} (valid: 0.0-1.0)")
    
    # Invalid claim (confidence out of bounds)
    print("\n❌ Attempting invalid claim (confidence > 1.0):")
    try:
        invalid_claim = Claim(
            claim_id="C2",
            text="Invalid claim",
            confidence=1.5  # Out of bounds!
        )
    except Exception as e:
        print(f"   Validation error caught: {type(e).__name__}")
        print(f"   ✅ Schema validation working correctly!")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("  VERIFICATION CONTROLLER DEMONSTRATION")
    print("  Step 1: Architecture Validation (No ML Models)")
    print("="*60)
    
    demo_basic_verification()
    demo_empty_answer()
    demo_custom_configuration()
    demo_schema_validation()
    
    print_section("Summary")
    print("✅ All demos completed successfully!")
    print("\n📋 What This Demonstrates:")
    print("   1. Clean architecture with structured schemas")
    print("   2. Type-safe validation with Pydantic")
    print("   3. Flexible configuration for different use cases")
    print("   4. Graceful handling of edge cases")
    print("   5. Production-ready logging and error handling")
    
    print("\n🎯 Next Steps:")
    print("   - Step 2: Integrate spaCy SVO claim extraction")
    print("   - Step 3: Add DeBERTa entailment verification")
    print("   - Step 4: Implement adaptive revision strategies")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
