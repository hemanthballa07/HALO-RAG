"""
Verification Controller Schemas
Defines the contract for claim-level verification and revision.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Claim(BaseModel):
    """
    Represents a single atomic claim extracted from generated text.
    """
    claim_id: str = Field(..., description="Unique identifier for this claim (e.g., 'C1', 'C2')")
    text: str = Field(..., description="The actual claim text")
    evidence_ids: List[str] = Field(default_factory=list, description="IDs of passages supporting this claim")
    entailment: str = Field(default="UNVERIFIED", description="Entailment label: ENTAILMENT, NEUTRAL, CONTRADICTION, or UNVERIFIED")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score for entailment (0.0 to 1.0)")


class VerificationResult(BaseModel):
    """
    Result of the verification process for a generated answer.
    """
    claims: List[Claim] = Field(..., description="List of extracted and verified claims")
    verified: bool = Field(..., description="Whether all claims are verified (entailed)")
    revision_cycles: int = Field(default=0, description="Number of revision cycles performed")
    reason: str = Field(..., description="Human-readable explanation of verification outcome")
    
    # Optional metadata
    total_claims: int = Field(default=0, description="Total number of claims extracted")
    verified_claims: int = Field(default=0, description="Number of claims with ENTAILMENT label")
    hallucinated_claims: int = Field(default=0, description="Number of claims with NEUTRAL or CONTRADICTION label")
    avg_confidence: float = Field(default=0.0, description="Average confidence across all claims")


class RevisionRequest(BaseModel):
    """
    Request for adaptive revision of a failed verification.
    """
    query: str = Field(..., description="Original user query")
    generated_answer: str = Field(..., description="Generated answer that failed verification")
    failed_claims: List[Claim] = Field(..., description="Claims that failed verification")
    retrieved_docs: dict = Field(..., description="Retrieved documents used for generation")
    strategy: Optional[str] = Field(default=None, description="Specific revision strategy to use (if None, auto-select)")
