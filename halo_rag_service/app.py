"""Small HTTP interface around the existing inference pipeline."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Callable, Literal

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    from src.pipeline import SelfVerificationRAGPipeline


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class GenerateRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k_retrieve: int = Field(default=20, gt=0, le=100)
    top_k_rerank: int = Field(default=5, gt=0, le=100)
    max_new_tokens: int = Field(default=256, gt=0, le=1024)
    temperature: float | None = Field(default=None, gt=0, le=2)
    do_sample: bool | None = None
    num_beams: int | None = Field(default=None, gt=0, le=20)
    max_revision_iterations: int | None = Field(default=None, ge=0, le=5)
    evidence_limit: int | None = Field(default=None, gt=0, le=100)
    abstain_if_unanswered: bool = False

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query cannot be blank")
        return value

    @model_validator(mode="after")
    def check_retrieval_depth(self) -> GenerateRequest:
        if self.top_k_rerank > self.top_k_retrieve:
            raise ValueError("top_k_rerank cannot exceed top_k_retrieve")
        return self


class Source(BaseModel):
    text: str


class VerificationSummary(BaseModel):
    verified: bool
    num_entailed: int
    num_total: int
    entailment_rate: float


class GenerateResponse(BaseModel):
    query: str
    answer: str
    status: Literal["verified", "unverified", "abstained"]
    verified: bool
    abstained: bool
    sources: list[Source]
    verification: VerificationSummary


def load_default_pipeline() -> SelfVerificationRAGPipeline:
    """Build one pipeline from the configured models and an explicit corpus file."""
    corpus_setting = os.environ.get("HALO_RAG_CORPUS_PATH")
    if not corpus_setting:
        raise RuntimeError("HALO_RAG_CORPUS_PATH must point to a corpus text file")
    corpus_path = Path(corpus_setting).expanduser()
    corpus = [line.strip() for line in corpus_path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if not corpus:
        raise ValueError(f"corpus is empty: {corpus_path}")

    config_path = Path(os.environ.get("HALO_RAG_CONFIG_PATH", PROJECT_ROOT / "config/config.yaml"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")

    retrieval = config.get("retrieval", {})
    fusion = retrieval.get("fusion", {})
    verification = config.get("verification", {})
    revision = config.get("revision", {})

    from src.pipeline import SelfVerificationRAGPipeline

    return SelfVerificationRAGPipeline(
        corpus=corpus,
        retrieval_model=retrieval.get("dense", {}).get(
            "model_name", "sentence-transformers/all-mpnet-base-v2"
        ),
        reranker_model=retrieval.get("reranker", {}).get(
            "model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        generator_model=config.get("generation", {}).get("model_name", "google/flan-t5-large"),
        verifier_model=verification.get("entailment_model", "cross-encoder/nli-deberta-v3-base"),
        entailment_threshold=verification.get("threshold", 0.75),
        dense_weight=fusion.get("dense_weight", 0.6),
        sparse_weight=fusion.get("sparse_weight", 0.4),
        device=config.get("experiments", {}).get("device", "auto"),
        use_qlora=False,
        max_revision_iterations=revision.get("max_iterations", 3),
        revision_config=revision,
    )


def create_app(
    pipeline_factory: Callable[[], SelfVerificationRAGPipeline] = load_default_pipeline,
) -> FastAPI:
    """Create an app that loads one pipeline at startup and shares it across requests."""
    inference_lock = Lock()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.pipeline = pipeline_factory()
        try:
            yield
        finally:
            application.state.pipeline = None

    application = FastAPI(title="HALO-RAG", lifespan=lifespan)

    @application.get("/health")
    def health() -> dict[str, str | int]:
        pipeline = application.state.pipeline
        return {"status": "ready", "device": pipeline.device, "corpus_size": len(pipeline.corpus)}

    @application.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        try:
            with inference_lock:
                result = application.state.pipeline.generate(**request.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Generation failed")
            raise HTTPException(status_code=500, detail="Generation failed") from exc

        try:
            verification = result["verification_results"]
            verified = bool(result["verified"])
            abstained = bool(result["abstained"])
            return GenerateResponse(
                query=result["query"],
                answer=result["generated_text"],
                status="abstained" if abstained else "verified" if verified else "unverified",
                verified=verified,
                abstained=abstained,
                sources=[Source(text=text) for text in result["reranked_texts"]],
                verification=VerificationSummary(
                    verified=bool(verification["verified"]),
                    num_entailed=int(verification["num_entailed"]),
                    num_total=int(verification["num_total"]),
                    entailment_rate=float(verification["entailment_rate"]),
                ),
            )
        except Exception as exc:
            logger.exception("Invalid pipeline result")
            raise HTTPException(status_code=500, detail="Generation failed") from exc

    return application


app = create_app()
