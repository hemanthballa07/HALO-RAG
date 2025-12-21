"""
Embedding service for text embedding generation.
"""

from typing import List
import numpy as np

from ..core.models import ModelLoader
from ..core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self, model_loader: ModelLoader):
        """
        Initialize embedding service.
        
        Args:
            model_loader: Model loader instance
        """
        self.model_loader = model_loader
    
    def generate_embeddings(self, texts: List[str]) -> tuple[List[List[float]], int]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            Tuple of (embeddings, dimension)
        """
        try:
            retriever = self.model_loader.get_retriever()
            
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Use the dense model from retriever
            embeddings = retriever.dense_model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=32
            )
            
            # Convert to list for JSON serialization
            embeddings_list = embeddings.tolist()
            dimension = retriever.embedding_dim
            
            logger.info(f"Generated {len(embeddings_list)} embeddings of dimension {dimension}")
            
            return embeddings_list, dimension
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}", exc_info=True)
            raise
