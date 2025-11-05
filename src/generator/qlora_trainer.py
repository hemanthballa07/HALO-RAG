"""
QLoRA Trainer for fine-tuning FLAN-T5 with QLoRA.
"""

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from typing import Dict, List, Optional
import os


class QLoRATrainer:
    """
    Trainer for fine-tuning FLAN-T5 with QLoRA.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: List[str] = None
    ):
        """
        Initialize QLoRA trainer.
        
        Args:
            model: Base model
            tokenizer: Tokenizer
            r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            target_modules: Target modules for LoRA (default: ["q", "v", "k", "o"])
        """
        self.model = model
        self.tokenizer = tokenizer
        
        if target_modules is None:
            target_modules = ["q", "v", "k", "o"]
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none"
        )
        
        # Add LoRA adapters
        self.model = get_peft_model(self.model, lora_config)
    
    def prepare_dataset(
        self,
        queries: List[str],
        contexts: List[str],
        answers: List[str],
        max_length: int = 512
    ) -> Dataset:
        """
        Prepare dataset for training.
        
        Args:
            queries: List of queries
            contexts: List of contexts
            answers: List of answers
            max_length: Maximum sequence length
        
        Returns:
            HuggingFace Dataset
        """
        # Format inputs: "Question: {query} Context: {context} Answer:"
        inputs = [
            f"Question: {q} Context: {c} Answer:"
            for q, c in zip(queries, contexts)
        ]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize targets (answers)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                answers,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        
        # Convert to dataset
        dataset = Dataset.from_dict({
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": model_inputs["labels"]
        })
        
        return dataset
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "./checkpoints",
        num_epochs: int = 3,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_strategy: str = "epoch",
        eval_strategy: str = "steps"
    ):
        """
        Train the model with QLoRA.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            output_dir: Output directory for checkpoints
            num_epochs: Number of training epochs
            batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            logging_steps: Logging frequency
            eval_steps: Evaluation frequency
            save_strategy: Save strategy ("epoch" or "steps")
            eval_strategy: Eval strategy ("epoch" or "steps")
        """
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            eval_steps=eval_steps if eval_dataset else None,
            save_strategy=save_strategy,
            eval_strategy=eval_strategy if eval_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            fp16=True,
            report_to="wandb" if os.getenv("WANDB_API_KEY") else None,
            run_name="flant5_qlora_rag"
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train
        trainer.train()
        
        # Save final model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer
    
    def save_checkpoint(self, path: str):
        """Save LoRA checkpoint."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

