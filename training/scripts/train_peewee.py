#!/usr/bin/env python3
# Author  : hackenstacks@protonmail.com
# Contact : hackenstacks@protonmail.com
"""
Peewee (NXS-Sovereign-Model) fine-tune
Base : huihui-ai/Huihui-Qwen3.5-4B-abliterated
Train: Unsloth QLoRA (or full fine-tune when VRAM >= 80GB)
Out  : GGUF Q4_K_M ready for Ollama

Usage:
  python train_peewee.py                   # QLoRA auto-detect
  python train_peewee.py --full            # full fine-tune (Blackwell 96GB)
  python train_peewee.py --epochs 5        # more epochs
  python train_peewee.py --data /path/to/data.jsonl
"""

import os
import argparse
import pathlib
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

SCRIPT_DIR  = pathlib.Path(__file__).parent
DATA_FILE   = SCRIPT_DIR.parent / "data"   / "nexus_train_mistral.jsonl"
OUTPUT_DIR  = SCRIPT_DIR.parent / "output" / "peewee-qlora"
GGUF_DIR    = SCRIPT_DIR.parent / "output" / "peewee-gguf"

BASE_MODEL     = "huihui-ai/Huihui-Qwen3.5-4B-abliterated"
MAX_SEQ_LEN    = 2048
LORA_TARGETS   = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default=str(DATA_FILE))
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--full",   action="store_true",
                        help="Full fine-tune — requires 80GB+ VRAM (Blackwell)")
    args = parser.parse_args()

    gpu_vram    = vram_gb()
    full_mode   = args.full and gpu_vram >= 80.0
    load_4bit   = not full_mode
    lr          = 3e-5 if full_mode else 2e-4
    batch       = 8    if full_mode else 4

    print(f"GPU VRAM  : {gpu_vram:.1f}GB")
    print(f"Mode      : {'Full fine-tune' if full_mode else 'QLoRA 4-bit'}")
    print(f"Dataset   : {args.data}")
    print(f"Epochs    : {args.epochs}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL,
        max_seq_length = MAX_SEQ_LEN,
        load_in_4bit   = load_4bit,
        dtype          = None,
    )

    if not full_mode:
        model = FastLanguageModel.get_peft_model(
            model,
            r                        = 16,
            lora_alpha               = 32,
            lora_dropout             = 0.05,
            target_modules           = LORA_TARGETS,
            bias                     = "none",
            use_gradient_checkpointing = "unsloth",
            random_state             = 42,
        )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    def apply_template(examples):
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in examples["messages"]
        ]
        return {"text": texts}

    dataset = load_dataset("json", data_files=args.data, split="train")
    dataset = dataset.map(apply_template, batched=True)

    os.makedirs(args.output, exist_ok=True)

    trainer = SFTTrainer(
        model        = model,
        tokenizer    = tokenizer,
        train_dataset = dataset,
        args = SFTConfig(
            output_dir                  = args.output,
            num_train_epochs            = args.epochs,
            per_device_train_batch_size = batch,
            gradient_accumulation_steps = 4,
            warmup_steps                = 100,
            learning_rate               = lr,
            fp16                        = not torch.cuda.is_bf16_supported(),
            bf16                        = torch.cuda.is_bf16_supported(),
            logging_steps               = 50,
            save_steps                  = 500,
            optim                       = "adamw_8bit",
            weight_decay                = 0.01,
            lr_scheduler_type           = "cosine",
            max_seq_length              = MAX_SEQ_LEN,
            dataset_text_field          = "text",
            report_to                   = "none",
        ),
    )

    print("\nTraining...")
    trainer.train()

    print("Saving adapter...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("Exporting GGUF Q4_K_M...")
    os.makedirs(GGUF_DIR, exist_ok=True)
    model.save_pretrained_gguf(
        str(GGUF_DIR / "peewee"),
        tokenizer,
        quantization_method = "q4_k_m",
    )

    for f in sorted(pathlib.Path(GGUF_DIR).glob("*.gguf")):
        size = f.stat().st_size / 1e9
        print(f"  {f.name}  {size:.2f}GB")

    print("\nDone.")
    print("Next: copy peewee-Q4_K_M.gguf to ~/NeXuS/models/ then run:")
    print("  ollama create peewee -f training/scripts/Modelfile")


if __name__ == "__main__":
    main()
