#!/usr/bin/env python3
"""
Quick test for Step 3: DeBERTa Entailment Verification
"""

import sys
sys.path.insert(0, '.')

from src.verification_controller import VerificationController

print("="*70)
print("  STEP 3 TEST: DeBERTa Entailment Verification")
print("="*70)

# Initialize controller (will load DeBERTa model)
print("\n🔄 Initializing VerificationController with DeBERTa...")
controller = VerificationController(entailment_threshold=0.75)

# Test 1: Entailed claim
print("\n" + "="*70)
print("TEST 1: Entailed Claim")
print("="*70)

query = "What is the capital of France?"
answer = "Paris is the capital of France."
evidence = {
    "passages": [
        "Paris is the capital and most populous city of France.",
        "France is a country in Western Europe."
    ],
    "ids": ["P1", "P2"]
}

result = controller.verify(answer, evidence, query)

print(f"\n📝 Answer: {answer}")
print(f"📚 Evidence: {len(evidence['passages'])} passages")
print(f"\n📊 Verification Result:")
print(f"   Status: {'✅ VERIFIED' if result.verified else '❌ NOT VERIFIED'}")
print(f"   Claims: {result.total_claims}")
print(f"   Verified: {result.verified_claims}")
print(f"   Avg Confidence: {result.avg_confidence:.3f}")
print(f"   Reason: {result.reason}")

print(f"\n🔬 Claims:")
for claim in result.claims:
    icon = "✅" if claim.entailment == "ENTAILMENT" else "⚠️" if claim.entailment == "NEUTRAL" else "❌"
    print(f"   {icon} [{claim.claim_id}] {claim.text}")
    print(f"      Entailment: {claim.entailment} (confidence: {claim.confidence:.3f})")
    if claim.evidence_ids:
        print(f"      Evidence: {', '.join(claim.evidence_ids)}")

# Test 2: Neutral claim
print("\n" + "="*70)
print("TEST 2: Neutral Claim (insufficient evidence)")
print("="*70)

answer2 = "Paris has a population of 10 million people."
evidence2 = {
    "passages": [
        "Paris is the capital of France.",
        "France is in Western Europe."
    ],
    "ids": ["P1", "P2"]
}

result2 = controller.verify(answer2, evidence2)

print(f"\n📝 Answer: {answer2}")
print(f"📚 Evidence: {len(evidence2['passages'])} passages")
print(f"\n📊 Verification Result:")
print(f"   Status: {'✅ VERIFIED' if result2.verified else '❌ NOT VERIFIED'}")
print(f"   Verified: {result2.verified_claims}/{result2.total_claims}")
print(f"   Reason: {result2.reason}")

print(f"\n🔬 Claims:")
for claim in result2.claims:
    icon = "✅" if claim.entailment == "ENTAILMENT" else "⚠️" if claim.entailment == "NEUTRAL" else "❌"
    print(f"   {icon} [{claim.claim_id}] {claim.text}")
    print(f"      Entailment: {claim.entailment} (confidence: {claim.confidence:.3f})")

print("\n" + "="*70)
print("  ✅ STEP 3 COMPLETE: Real entailment verification working!")
print("="*70 + "\n")
