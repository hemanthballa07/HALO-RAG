"""
Configuration management using Pydantic Settings.
Loads from environment variables and YAML config.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field
import yaml


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Service Configuration
    app_name: str = "HALO-RAG Service"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    
    # Device Configuration
    device: str = Field(default="cuda", description="Device for model inference (cuda/cpu)")
    
    # Model Paths
    dense_model_name: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Dense retrieval model"
    )
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder reranker model"
    )
    generator_model_name: str = Field(
        default="google/flan-t5-large",
        description="Generator model"
    )
    entailment_model_name: str = Field(
        default="cross-encoder/nli-deberta-v3-base",
        description="Entailment verification model"
    )
    generator_lora_checkpoint: Optional[str] = Field(
        default=None,
        description="Path to LoRA checkpoint for generator"
    )
    
    # Retrieval Configuration
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    top_k_retrieve: int = Field(default=20, gt=0, description="Number of documents to retrieve")
    top_k_rerank: int = Field(default=5, gt=0, description="Number of documents after reranking")
    
    # Generation Configuration
    max_new_tokens: int = Field(default=256, gt=0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=50, gt=0)
    do_sample: bool = Field(default=True)
    num_beams: int = Field(default=1, gt=0)
    
    # Verification Configuration
    entailment_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    enable_verification: bool = Field(default=True)
    enable_revision: bool = Field(default=True)
    max_revision_iterations: int = Field(default=3, ge=0)
    
    # QLoRA Configuration
    use_qlora: bool = Field(default=True)
    lora_r: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    lora_dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Corpus Configuration
    corpus_path: Optional[str] = Field(
        default=None,
        description="Path to corpus file (one document per line)"
    )
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json/text)")
    
    # YAML Config Path
    config_yaml_path: Optional[str] = Field(
        default=None,
        description="Path to YAML config file (overrides defaults)"
    )
    
    class Config:
        env_prefix = "HALO_RAG_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Load from YAML if path provided
        if self.config_yaml_path and os.path.exists(self.config_yaml_path):
            self._load_from_yaml(self.config_yaml_path)
    
    def _load_from_yaml(self, yaml_path: str):
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Map YAML structure to settings
        if 'retrieval' in config_data:
            if 'dense' in config_data['retrieval']:
                self.dense_model_name = config_data['retrieval']['dense'].get(
                    'model_name', self.dense_model_name
                )
            if 'fusion' in config_data['retrieval']:
                self.dense_weight = config_data['retrieval']['fusion'].get(
                    'dense_weight', self.dense_weight
                )
                self.sparse_weight = config_data['retrieval']['fusion'].get(
                    'sparse_weight', self.sparse_weight
                )
            if 'reranker' in config_data['retrieval']:
                self.reranker_model_name = config_data['retrieval']['reranker'].get(
                    'model_name', self.reranker_model_name
                )
        
        if 'generation' in config_data:
            self.generator_model_name = config_data['generation'].get(
                'model_name', self.generator_model_name
            )
            self.max_new_tokens = config_data['generation'].get(
                'max_new_tokens', self.max_new_tokens
            )
            self.temperature = config_data['generation'].get(
                'temperature', self.temperature
            )
        
        if 'verification' in config_data:
            self.entailment_model_name = config_data['verification'].get(
                'entailment_model', self.entailment_model_name
            )
            self.entailment_threshold = config_data['verification'].get(
                'threshold', self.entailment_threshold
            )
        
        if 'experiments' in config_data:
            self.device = config_data['experiments'].get('device', self.device)


# Global settings instance
settings = Settings()
