"""
Fine-tune Bubble Transformer with LoRA
======================================

LoRA (Low-Rank Adaptation) fine-tuning for Bubble Transformer V3.
Freezes base model and trains only low-rank adapters.

Usage:
python scripts/finetune_bubble.py --model Qwen/Qwen2.5-0.5B --epochs 3 --lr 1e-4

Requirements:
pip install peft datasets
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def check_dependencies():
    """Check if required packages are installed."""
    try:
        import peft
        import datasets

        return True
    except ImportError as e:
        print(f"\n[ERROR] Missing dependency: {e}")
        print("\nInstall required packages:")
        print("  pip install peft datasets")
        return False


def load_model_with_bubble(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    target_layers: List[int] = [3, 7, 11],
    thresholds_path: str = "calibración/layer_thresholds_progressive.json",
    device: str = "cuda",
):
    """Load model with Bubble Transformer injected."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts.generate_with_bubble import inject_bubble_attention
    import json

    print(f"\n{'=' * 60}")
    print(f"Loading {model_name}...")
    print(f"{'=' * 60}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to(device)

    # Load thresholds
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r") as f:
            thresholds = json.load(f)
        print(f"[Info] Loaded calibration thresholds from {thresholds_path}")
    else:
        thresholds = {}
        print(f"[Warning] Thresholds file not found: {thresholds_path}")

    # Inject Bubble Attention
    model = inject_bubble_attention(
        model=model,
        target_layers=target_layers,
        thresholds=thresholds,
        default_C=32,
    )

    return model, tokenizer


def prepare_dataset(
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    max_length: int = 512,
    num_samples: int = 1000,
):
    """Prepare dataset for fine-tuning."""
    from datasets import load_dataset

    print(f"\n{'=' * 60}")
    print("Preparing Dataset...")
    print(f"{'=' * 60}")
    print(f"Dataset: {dataset_name}/{dataset_config}")
    print(f"Max length: {max_length}")
    print(f"Num samples: {num_samples}")

    # Load dataset
    try:
        dataset = load_dataset(dataset_name, dataset_config, split="train")
    except Exception as e:
        print(f"[Warning] Failed to load {dataset_name}: {e}")
        print("[Info] Using synthetic dataset instead")

        # Create synthetic dataset
        texts = [
            "El Bubble Transformer es una arquitectura innovadora que reemplaza la atención tradicional."
            for _ in range(num_samples)
        ]
        dataset = type("obj", (object,), {"data": texts})()

    # Tokenize
    def tokenize_function(examples):
        # Handle different dataset formats
        if isinstance(examples, dict) and "text" in examples:
            texts = examples["text"]
        else:
            texts = examples if isinstance(examples, list) else [examples]

        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    # Apply tokenization
    try:
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
            if hasattr(dataset, "column_names")
            else None,
        )
    except Exception as e:
        print(f"[Warning] Dataset mapping failed: {e}")
        print("[Info] Creating minimal dataset")

        # Create minimal dataset
        from torch.utils.data import Dataset

        class MinimalDataset(Dataset):
            def __init__(self, tokenizer, num_samples=100):
                self.tokenizer = tokenizer
                self.num_samples = num_samples
                self.text = "El Bubble Transformer optimiza la atención mediante transporte óptimo."

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                return {
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "labels": encoding["input_ids"].squeeze(0).clone(),
                }

        tokenized_dataset = MinimalDataset(tokenizer, num_samples=num_samples)

    print(f"Dataset size: {len(tokenized_dataset)}")

    return tokenized_dataset


def apply_lora(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: List[str] = ["W_q", "W_k", "W_v", "W_o"],
):
    """Apply LoRA to model."""
    from peft import LoraConfig, get_peft_model

    print(f"\n{'=' * 60}")
    print("Applying LoRA...")
    print(f"{'=' * 60}")
    print(f"LoRA rank: {r}")
    print(f"LoRA alpha: {lora_alpha}")
    print(f"LoRA dropout: {lora_dropout}")
    print(f"Target modules: {target_modules}")

    # LoRA config
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params

    print(f"\nTrainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)")
    print(f"Total parameters: {total_params:,}")

    return model


def finetune_bubble(
    model: nn.Module,
    tokenizer,
    dataset,
    output_dir: str = "./bubble_finetuned",
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    device: str = "cuda",
):
    """Fine-tune model with LoRA."""
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

    print(f"\n{'=' * 60}")
    print("Fine-tuning Bubble Transformer...")
    print(f"{'=' * 60}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        logging_dir=f"{output_dir}/logs",
        report_to="none",
        disable_tqdm=False,
        fp16=device == "cuda",
        optim="adamw_torch",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n[Training] Starting...")
    try:
        trainer.train()
        print("\n[Training] Completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        raise

    # Save
    print(f"\n[Saving] Model saved to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Bubble Transformer V3 with LoRA"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model name or path",
    )
    parser.add_argument(
        "--target-layers",
        type=int,
        nargs="+",
        default=[3, 7, 11],
        help="Target layer indices for injection",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="calibración/layer_thresholds_progressive.json",
        help="Path to calibration thresholds JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./bubble_finetuned",
        help="Output directory for fine-tuned model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per device",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of training samples",
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        return

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\n[Warning] CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load model with Bubble Transformer
    model, tokenizer = load_model_with_bubble(
        model_name=args.model,
        target_layers=args.target_layers,
        thresholds_path=args.thresholds,
        device=args.device,
    )

    # Prepare dataset
    dataset = prepare_dataset(
        tokenizer=tokenizer,
        num_samples=args.num_samples,
    )

    # Apply LoRA
    model = apply_lora(
        model=model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    # Fine-tune
    model = finetune_bubble(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"\n{'=' * 60}")
    print("FINE-TUNING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Model saved to: {args.output_dir}")
    print("\nTo use the fine-tuned model:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.output_dir}')")


if __name__ == "__main__":
    main()
