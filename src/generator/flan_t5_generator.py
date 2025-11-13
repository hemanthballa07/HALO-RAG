"""
FLAN-T5 Generator Module with QLoRA fine-tuning support.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from typing import List, Optional, Dict, Any


class FLANT5Generator:
    """
    FLAN-T5 generator with QLoRA fine-tuning support.
    Uses 4-bit NF4 quantization with r=16 LoRA.
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        device: str = "cuda",
        lora_checkpoint: Optional[str] = None,
        use_qlora: bool = True,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        bits: int = 4,
        bit_type: str = "nf4"
    ):
        """
        Initialize FLAN-T5 generator.
        
        Args:
            model_name: Base model name
            device: Device to run model on
            lora_checkpoint: Path to LoRA checkpoint (if fine-tuned)
            use_qlora: Whether to use QLoRA (4-bit quantization)
            r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout
            bits: Quantization bits (4 for QLoRA)
            bit_type: Quantization type ("nf4" or "fp4")
        """
        self.device = device
        self.model_name = model_name
        self.use_qlora = use_qlora
        
        # Configure quantization if using QLoRA
        # QLoRA requires CUDA - disable on CPU/MPS
        if use_qlora and device == "cuda":
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=bit_type,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
            except Exception as e:
                print(f"⚠ Warning: QLoRA initialization failed: {e}. Falling back to full precision.")
                use_qlora = False
                quantization_config = None
        else:
            if use_qlora and device != "cuda":
                print(f"⚠ QLoRA requires CUDA but device is {device}. Disabling QLoRA.")
            use_qlora = False
            quantization_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load base model
        if use_qlora and quantization_config is not None:
            # QLoRA path (CUDA only)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            # Full precision path (CPU/MPS or no QLoRA)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name
            )
            self.model.to(device)
        
        # Configure LoRA
        if lora_checkpoint:
            # Load fine-tuned LoRA weights
            # For iterative training, keep as PEFT model (don't merge)
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_checkpoint
            )
            # Don't merge - keep as PEFT model for further fine-tuning
        else:
            # Set up LoRA configuration
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=r,
                lora_alpha=lora_alpha,
                target_modules=["q", "v", "k", "o"],
                lora_dropout=lora_dropout,
                bias="none"
            )
            
            # Add LoRA adapters (but don't load checkpoint)
            # Only add if we're planning to fine-tune
            if use_qlora:
                self.model = get_peft_model(self.model, lora_config)
        
        self.model.eval()
    
    def generate(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = True,
        num_beams: int = 1
    ) -> str:
        """
        Generate answer given query and context.
        
        Args:
            query: Query string
            context: Retrieved context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            do_sample: Whether to use sampling
            num_beams: Number of beams for beam search
        
        Returns:
            Generated text
        """
        # Format input: "Question: {query} Context: {context} Answer:"
        input_text = f"Question: {query} Context: {context} Answer:"
        
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if do_sample:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def generate_batch(
        self,
        queries: List[str],
        contexts: List[str],
        **generation_kwargs
    ) -> List[str]:
        """
        Generate answers for multiple queries.
        
        Args:
            queries: List of queries
            contexts: List of contexts
            **generation_kwargs: Additional generation parameters
        
        Returns:
            List of generated texts
        """
        results = []
        for query, context in zip(queries, contexts):
            generated = self.generate(query, context, **generation_kwargs)
            results.append(generated)
        
        return results
    
    def generate_with_verification_hint(
        self,
        query: str,
        context: str,
        verified_claims: Optional[List[str]] = None,
        **generation_kwargs
    ) -> str:
        """
        Generate with verification hints to guide factual generation.
        
        Args:
            query: Query string
            context: Retrieved context
            verified_claims: List of verified claims to support
            **generation_kwargs: Generation parameters
        
        Returns:
            Generated text
        """
        # Add verified claims to context if provided
        if verified_claims:
            verified_text = " Verified facts: " + " | ".join(verified_claims)
            context = context + verified_text
        
        return self.generate(query, context, **generation_kwargs)

