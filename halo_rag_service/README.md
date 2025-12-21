# HALO-RAG Production Service

**FastAPI-based ML Inference Service for Self-Verification RAG**

A production-ready machine learning service that wraps a research-grade Retrieval-Augmented Generation (RAG) pipeline with entailment-based verification and adaptive revision strategies.

---

## 🎯 Overview

This service transforms a research-quality Self-Verification RAG system into a production ML inference API suitable for enterprise deployment. It demonstrates **production ML engineering** skills by:

- **Separating concerns** between API, service, and ML layers
- **Loading models once** at startup using singleton pattern
- **Type-safe validation** with Pydantic schemas
- **Structured logging** for observability and debugging
- **Comprehensive testing** with mocked and real model tests
- **Graceful error handling** and clear failure modes

### Core ML Pipeline

The underlying ML pipeline includes:

1. **Hybrid Retrieval**: FAISS (dense) + BM25 (sparse) fusion (0.6/0.4 weights)
2. **Cross-Encoder Reranking**: DeBERTa-v3 on MS MARCO
3. **Fine-Tuned Generation**: FLAN-T5-Large with QLoRA (r=16, 4-bit NF4)
4. **Entailment Verification**: DeBERTa-v3-large (MNLI + FEVER) with spaCy claim extraction
5. **Adaptive Revision**: Re-retrieval, constrained generation, claim-by-claim strategies

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ /health  │  │  /embed  │  │/retrieve │  │/generate │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└───────┼─────────────┼─────────────┼─────────────┼──────────┘
        │             │             │             │
┌───────┼─────────────┼─────────────┼─────────────┼──────────┐
│       │      Service Layer        │             │          │
│       │    ┌──────────────┐  ┌────┴────────┐  ┌┴────────┐ │
│       │    │  Embedding   │  │  Retrieval  │  │   RAG   │ │
│       │    │   Service    │  │   Service   │  │ Service │ │
│       │    └──────────────┘  └─────────────┘  └─────────┘ │
└───────┼──────────────────────────────────────────┼─────────┘
        │                                          │
┌───────┼──────────────────────────────────────────┼─────────┐
│       │           ML Layer (Research Code)       │         │
│  ┌────┴─────┐  ┌──────────────┐  ┌──────────────┴──────┐  │
│  │  Model   │  │    Hybrid    │  │  Self-Verification  │  │
│  │  Loader  │  │  Retriever   │  │   RAG Pipeline      │  │
│  │(Singleton)│  │  + Reranker  │  │  (Full Pipeline)    │  │
│  └──────────┘  └──────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

- **Separation of Concerns**: API routes → Services → ML components
- **Dependency Injection**: Models loaded once, injected via FastAPI
- **Fail-Safe**: Graceful degradation with clear error messages
- **Observability**: Structured JSON logging with request tracing

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM (for model loading)

### Installation

```bash
# Navigate to service directory
cd halo_rag_service

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For testing

# Download spaCy model
python -m spacy download en_core_web_sm

# Configure environment (optional)
cp .env.example .env
# Edit .env with your settings
```

### Running the Service

```bash
# Start the service
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development
uvicorn app.main:app --reload
```

The service will:
1. Load all ML models at startup (takes 2-5 minutes)
2. Build retrieval index from corpus
3. Start accepting requests at `http://localhost:8000`

### Verify It's Running

```bash
# Health check
curl http://localhost:8000/health

# View API documentation
open http://localhost:8000/docs
```

---

## 📡 API Reference

### 1. Health Check

**GET** `/health`

Check service status and model loading state.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "corpus_size": 1000,
  "device": "cuda"
}
```

---

### 2. Generate Embeddings

**POST** `/embed`

Generate dense embeddings for text inputs.

**Request:**
```json
{
  "texts": ["What is retrieval-augmented generation?"]
}
```

**Response:**
```json
{
  "embeddings": [[0.123, -0.456, ...]],
  "dimension": 768,
  "num_texts": 1
}
```

**Use Cases:**
- Semantic search
- Document similarity
- Clustering

---

### 3. Retrieve and Rerank

**POST** `/retrieve`

Retrieve documents using hybrid retrieval and rerank with cross-encoder.

**Request:**
```json
{
  "query": "What is the capital of France?",
  "top_k": 20,
  "rerank_k": 5
}
```

**Response:**
```json
{
  "query": "What is the capital of France?",
  "passages": [
    {
      "doc_id": 0,
      "text": "Paris is the capital of France.",
      "score": 0.95
    }
  ],
  "retrieval_time_ms": 123.45,
  "reranking_time_ms": 45.67
}
```

**Use Cases:**
- Document retrieval
- Evidence gathering
- Context selection for generation

---

### 4. RAG Generation (Full Pipeline)

**POST** `/generate`

Generate answer using full RAG pipeline with verification and revision.

**Request:**
```json
{
  "query": "When was the Eiffel Tower built?",
  "top_k_retrieve": 20,
  "top_k_rerank": 5,
  "enable_verification": true,
  "enable_revision": true,
  "temperature": 0.7,
  "max_new_tokens": 256
}
```

**Response:**
```json
{
  "query": "When was the Eiffel Tower built?",
  "answer": "The Eiffel Tower was built from 1887 to 1889.",
  "status": "VERIFIED",
  "sources": [
    {
      "doc_id": 1,
      "text": "The Eiffel Tower was constructed from 1887 to 1889.",
      "score": 0.95
    }
  ],
  "verification": {
    "verified": true,
    "total_claims": 1,
    "verified_claims": 1,
    "hallucinated_claims": 0,
    "avg_confidence": 0.92,
    "revision_cycles": 0,
    "reason": "All claims verified",
    "claims": []
  },
  "retrieval_time_ms": 123.45,
  "generation_time_ms": 234.56,
  "verification_time_ms": 45.67,
  "total_time_ms": 403.68
}
```

**Status Values:**
- `VERIFIED`: Answer passed verification
- `ABSTAINED`: Answer failed verification (returns safe fallback)

**Use Cases:**
- Question answering
- Factual information retrieval
- Hallucination-free generation

---

## ⚙️ Configuration

### Environment Variables

Set via `.env` file or environment:

```bash
# Service
HALO_RAG_DEBUG=false
HALO_RAG_PORT=8000

# Device
HALO_RAG_DEVICE=cuda  # or cpu

# Models
HALO_RAG_GENERATOR_LORA_CHECKPOINT=/path/to/checkpoint

# Retrieval
HALO_RAG_TOP_K_RETRIEVE=20
HALO_RAG_TOP_K_RERANK=5

# Verification
HALO_RAG_ENTAILMENT_THRESHOLD=0.75
HALO_RAG_ENABLE_VERIFICATION=true

# Logging
HALO_RAG_LOG_LEVEL=INFO
HALO_RAG_LOG_FORMAT=json
```

### Using Existing YAML Config

```bash
HALO_RAG_CONFIG_YAML_PATH=../config/config.yaml
```

---

## 🧪 Testing

### Run Unit Tests (Fast)

```bash
# Run all unit tests with mocked models
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=app --cov-report=html
```

**Unit tests:**
- ✅ Mock heavy models (no GPU needed)
- ✅ Test API validation
- ✅ Test error handling
- ✅ Fast execution (< 5 seconds)

### Run Integration Tests (Slow)

```bash
# Run integration tests with real models
pytest tests/integration/ -v -m integration
```

**Integration tests:**
- ⚠️ Load real models (requires GPU)
- ⚠️ Slow execution (minutes)
- ✅ End-to-end validation

### Skip Integration Tests

```bash
pytest -m "not integration"
```

---

## 📊 Logging & Observability

### Structured Logging

All logs are JSON-formatted for easy parsing:

```json
{
  "timestamp": "2025-12-21T05:30:00Z",
  "level": "INFO",
  "logger": "app.services.rag_service",
  "message": "Pipeline completed in 403.68ms",
  "request_id": "abc-123-def",
  "extra": {
    "total_time_ms": 403.68,
    "retrieval_time_ms": 123.45,
    "generation_time_ms": 234.56,
    "verification_time_ms": 45.67,
    "status": "VERIFIED"
  }
}
```

### Request Tracing

Each request gets a unique ID for tracking:

```bash
curl -H "X-Request-ID: my-custom-id" http://localhost:8000/generate
```

### Key Metrics Logged

- **Retrieval**: scores, timing, number of documents
- **Verification**: claims verified/hallucinated, confidence, revision cycles
- **Performance**: latency per pipeline stage
- **Errors**: stack traces, input validation failures

---

## 🐳 Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy application
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Run service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t halo-rag-service .
docker run -p 8000:8000 --gpus all halo-rag-service
```

### Production Considerations

1. **Model Caching**: Mount model cache as volume to avoid re-downloading
2. **GPU Access**: Use `--gpus all` for CUDA support
3. **Memory**: Allocate 16GB+ RAM
4. **Timeouts**: Set appropriate timeouts for model loading (5+ minutes)
5. **Health Checks**: Use `/health` endpoint for readiness probes

---

## 💼 Interview Talking Points

### What This Demonstrates

**1. Production ML Engineering**
- Model serving with FastAPI
- Singleton pattern for efficient model loading
- Separation of ML logic from API layer

**2. Software Engineering Best Practices**
- Type safety with Pydantic
- Dependency injection
- Comprehensive error handling
- Clean architecture (API → Service → ML)

**3. Testing Strategy**
- Unit tests with mocked models (fast CI/CD)
- Integration tests with real models (validation)
- Clear test organization

**4. Observability & Debugging**
- Structured JSON logging
- Request tracing with IDs
- Performance metrics per pipeline stage
- Verification metrics (claims, confidence, revisions)

**5. Collaboration Readiness**
- API-first design
- Clear documentation
- Configurable via environment variables
- Testable components

### Key Achievements

✅ **Zero changes to research code** - All ML logic preserved  
✅ **Models load once** - Fast inference, memory efficient  
✅ **Type-safe** - Pydantic validation prevents runtime errors  
✅ **Observable** - Structured logs for debugging  
✅ **Testable** - Mocked tests for fast iteration  
✅ **Production-ready** - Error handling, health checks, CORS

---

## 📁 Project Structure

```
halo_rag_service/
├── app/
│   ├── main.py                 # FastAPI app with lifespan events
│   ├── api/
│   │   ├── routes.py           # API endpoint definitions
│   │   └── dependencies.py     # Dependency injection
│   ├── core/
│   │   ├── config.py           # Configuration management
│   │   ├── logging.py          # Structured logging
│   │   └── models.py           # Model loader singleton
│   ├── schemas/
│   │   ├── requests.py         # Request validation schemas
│   │   └── responses.py        # Response schemas
│   └── services/
│       ├── embedding_service.py    # Embedding generation
│       ├── retrieval_service.py    # Retrieval + reranking
│       └── rag_service.py          # Full RAG pipeline
├── tests/
│   ├── conftest.py             # Pytest fixtures
│   ├── unit/
│   │   └── test_api.py         # API endpoint tests
│   └── integration/
│       └── test_pipeline.py    # End-to-end tests
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── .env.example                # Environment variables template
├── pytest.ini                  # Pytest configuration
└── README.md                   # This file
```

---

## 🤝 Contributing

This is a production wrapper for a research project. To modify:

1. **API changes**: Edit `app/api/routes.py`
2. **Service logic**: Edit `app/services/*.py`
3. **ML pipeline**: Edit parent `src/` directory (research code)
4. **Configuration**: Edit `app/core/config.py`

---

## 📝 License

Academic use only - University of Florida CIS 6930

---

## 🙏 Acknowledgments

Built on top of the HALO-RAG research pipeline implementing:
- Self-Verification Chains for Hallucination-Free RAG
- Hybrid retrieval with cross-encoder reranking
- Fine-tuned generation with QLoRA
- Entailment-based verification with adaptive revision

**For Recruiters**: This project demonstrates production ML engineering skills including model serving, API design, testing, and observability - all critical for MLE roles at companies like Microsoft.
