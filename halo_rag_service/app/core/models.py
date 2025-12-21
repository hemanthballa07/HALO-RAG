"""
Model loader singleton for efficient model management.
Loads models once at startup and provides thread-safe access.
"""

import sys
import os
from pathlib import Path
from typing import Optional, List
import logging
from threading import Lock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.retrieval import HybridRetriever, CrossEncoderReranker
from src.generator import FLANT5Generator
from src.verification_controller import VerificationController
from src.pipeline import SelfVerificationRAGPipeline

from .config import settings
from .logging import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """
    Singleton model loader for efficient model management.
    Models are loaded once and cached for reuse.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize model loader (only once)."""
        if self._initialized:
            return
        
        self._initialized = True
        self._models_loaded = False
        
        # Model instances
        self.retriever: Optional[HybridRetriever] = None
        self.reranker: Optional[CrossEncoderReranker] = None
        self.generator: Optional[FLANT5Generator] = None
        self.verification_controller: Optional[VerificationController] = None
        self.rag_pipeline: Optional[SelfVerificationRAGPipeline] = None
        
        # Corpus
        self.corpus: Optional[List[str]] = None
    
    def load_models(self, corpus: Optional[List[str]] = None) -> None:
        """
        Load all models at startup.
        
        Args:
            corpus: Document corpus for retrieval (optional, can be loaded from file)
        """
        if self._models_loaded:
            logger.info("Models already loaded, skipping")
            return
        
        logger.info("Starting model loading...")
        
        try:
            # Load corpus
            self.corpus = self._load_corpus(corpus)
            logger.info(f"Loaded corpus with {len(self.corpus)} documents")
            
            # Load retriever
            logger.info(f"Loading hybrid retriever (device: {settings.device})...")
            self.retriever = HybridRetriever(
                dense_model_name=settings.dense_model_name,
                dense_weight=settings.dense_weight,
                sparse_weight=settings.sparse_weight,
                device=settings.device
            )
            
            # Build retrieval index
            logger.info("Building retrieval index...")
            self.retriever.build_index(self.corpus)
            logger.info("Retrieval index built successfully")
            
            # Load reranker
            logger.info(f"Loading reranker ({settings.reranker_model_name})...")
            self.reranker = CrossEncoderReranker(
                model_name=settings.reranker_model_name,
                device=settings.device
            )
            logger.info("Reranker loaded successfully")
            
            # Load generator
            logger.info(f"Loading generator ({settings.generator_model_name})...")
            self.generator = FLANT5Generator(
                model_name=settings.generator_model_name,
                device=settings.device,
                lora_checkpoint=settings.generator_lora_checkpoint,
                use_qlora=settings.use_qlora,
                r=settings.lora_r,
                lora_alpha=settings.lora_alpha,
                lora_dropout=settings.lora_dropout
            )
            logger.info("Generator loaded successfully")
            
            # Load verification controller
            logger.info("Loading verification controller...")
            self.verification_controller = VerificationController(
                entailment_threshold=settings.entailment_threshold,
                max_revision_cycles=settings.max_revision_iterations,
                enable_revision=settings.enable_revision
            )
            logger.info("Verification controller loaded successfully")
            
            # Initialize RAG pipeline
            logger.info("Initializing RAG pipeline...")
            self.rag_pipeline = SelfVerificationRAGPipeline(
                corpus=self.corpus,
                retrieval_model=settings.dense_model_name,
                reranker_model=settings.reranker_model_name,
                generator_model=settings.generator_model_name,
                entailment_model=settings.entailment_model_name,
                device=settings.device,
                use_qlora=settings.use_qlora,
                generator_lora_checkpoint=settings.generator_lora_checkpoint,
                enable_revision=settings.enable_revision,
                max_revision_iterations=settings.max_revision_iterations
            )
            logger.info("RAG pipeline initialized successfully")
            
            self._models_loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}", exc_info=True)
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def _load_corpus(self, corpus: Optional[List[str]] = None) -> List[str]:
        """
        Load corpus from file or use provided corpus.
        
        Args:
            corpus: Optional corpus to use
        
        Returns:
            List of documents
        """
        if corpus is not None:
            return corpus
        
        if settings.corpus_path and os.path.exists(settings.corpus_path):
            logger.info(f"Loading corpus from {settings.corpus_path}")
            with open(settings.corpus_path, 'r', encoding='utf-8') as f:
                corpus = [line.strip() for line in f if line.strip()]
            return corpus
        
        # Default demo corpus
        logger.warning("No corpus provided, using demo corpus")
        return [
            "Paris is the capital and most populous city of France.",
            "The Eiffel Tower was constructed from 1887 to 1889.",
            "The Seine is a river that flows through Paris.",
            "France is a country in Western Europe.",
            "Gustave Eiffel designed the tower for the 1889 World's Fair.",
            "The Louvre Museum is located in Paris.",
            "French is the official language of France.",
            "The French Revolution began in 1789.",
            "Napoleon Bonaparte was a French military leader.",
            "The Palace of Versailles is a former royal residence near Paris."
        ]
    
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._models_loaded
    
    def get_retriever(self) -> HybridRetriever:
        """Get retriever instance."""
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self.retriever
    
    def get_reranker(self) -> CrossEncoderReranker:
        """Get reranker instance."""
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self.reranker
    
    def get_generator(self) -> FLANT5Generator:
        """Get generator instance."""
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self.generator
    
    def get_verification_controller(self) -> VerificationController:
        """Get verification controller instance."""
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self.verification_controller
    
    def get_rag_pipeline(self) -> SelfVerificationRAGPipeline:
        """Get RAG pipeline instance."""
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self.rag_pipeline
    
    def get_corpus(self) -> List[str]:
        """Get corpus."""
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self.corpus


# Global model loader instance
_model_loader = None


def get_model_loader() -> ModelLoader:
    """Get global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader
