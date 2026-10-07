# Peewee — NXS-Sovereign-Model

**A sovereign AI fine-tuned on the NeXuS platform stack.**

Peewee is a domain-specific language model fine-tuned from [Huihui Qwen3.5-4B Abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-4B-abliterated) on 64,250 samples of NeXuS operational knowledge — darknet services, Alpine Linux administration, security hardening, DivaChain, NeXiuM, sovereign infrastructure, and agentic coding traces.

The output is a GGUF Q4_K_M (~2.5GB) that runs locally in Ollama with no external API dependency.

> *Sane • Simple • Secure • Stealthy • Beautiful — NeXuS Principle*

**Author:** hackenstacks@protonmail.com
**Contact:** hackenstacks@protonmail.com
**License:** Apache 2.0 (inherited from base model)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Quick Start — Google Colab / Antigravity](#quick-start--google-colab--antigravity)
- [Quick Start — Local GPU](#quick-start--local-gpu)
- [Deploy with Ollama](#deploy-with-ollama)
- [Repository Structure](#repository-structure)
- [Training Parameters](#training-parameters)
- [Hardware Reference](#hardware-reference)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

---

## Overview

Peewee is **Tier 3** in the NeXuS AI stack — the local domain expert:

```
Tier 0  TeleOCR 1.2B     — always-on vision/OCR/PDF (700MB)
Tier 1  Sky-T1 1B GRPO   — reasoning (500MB)
Tier 2  Huihui 4B        — general uncensored assistant (3.3GB)
Tier 3  Peewee (this)    — NeXuS domain expert, fine-tuned (2.5GB)
Tier 4  HuggingFace      — cloud inference fallback
Tier 5  Merlin/Gemini    — frontier API (Ghost Gate permit required)
```

Routing is handled by `nexus-ai-router` at `localhost:8810`. All tiers run locally — Tier 4/5 require explicit authorization.

---

## Architecture

| Property | Value |
|----------|-------|
| Base model | `huihui-ai/Huihui-Qwen3.5-4B-abliterated` |
| Architecture | Qwen3.5 4B |
| Fine-tune method | QLoRA (LoRA r=16, alpha=32) |
| Full fine-tune | Supported on 80GB+ VRAM (Blackwell) |
| Output format | GGUF Q4_K_M |
| Output size | ~2.5GB |
| Context length | 4096 tokens (inference) / 2048 (training) |
| Chat template | Qwen-2.5 |
| Trainer | [Unsloth](https://github.com/unslothai/unsloth) + TRL SFTTrainer |

**Why Huihui Qwen3.5-4B Abliterated?**
- Apache 2.0 license — weights returnable, commercially usable
- Abliterated (6/100 refusals) — no artificial restrictions for sovereign operation
- Vision confirmed working
- Native Ollama support
- Unsloth QLoRA compatible

---

## Dataset

The training dataset (`nexus_train_mistral.jsonl`) is **not stored in this repo** due to size. Download from the [latest release](https://github.com/hackenstacks/peewee/releases).

| Property | Value |
|----------|-------|
| Samples | 64,250 |
| Format | Mistral messages JSONL |
| Size | 44MB |
| Bad samples | 0 (validated) |

**Dataset sources:**

| Source | Description | Samples |
|--------|-------------|---------|
| NeXuS docs | Architecture, whitepapers, design docs | ~8,000 |
| Arena corpus | 174 Claude conversation exports | ~12,000 |
| TLDR pages | 7,492 Linux/Unix command references | ~15,000 |
| Cheatsheets | 35+ bash/python/go/tmux/vim/ssh notes | ~2,000 |
| Fable-10K | Agentic coding traces (reasoning extraction) | ~4,400 |
| NeXuS scripts | nexus-*.sh operational knowledge | ~22,850 |

**Format (every sample):**
```json
{
  "messages": [
    {"role": "system",    "content": "You are NEXUS-AI, an expert assistant for the NeXuS sovereign platform..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Validate the dataset before training:**
```bash
python3 training/scripts/validate_data.py
```

---

## Requirements

### Python dependencies

```bash
pip install -r training/scripts/requirements.txt
```

Contents:
```
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
trl>=0.17.0
datasets>=3.0.0
transformers>=4.47.0
accelerate>=1.0.0
peft>=0.13.0
bitsandbytes>=0.45.0
torch>=2.5.0
```

### GPU requirements

| Mode | Min VRAM | Recommended | Est. time (64K samples) |
|------|----------|-------------|------------------------|
| QLoRA 4-bit | 10GB | T4 (15GB) | 2–4 hours |
| QLoRA 4-bit | 10GB | A100 (40GB) | ~45 min |
| Full fine-tune | 80GB | RTX Pro 6000 Blackwell (96GB) | ~15–30 min |

> **No GPU?** The script exits gracefully on CPU — training is not feasible locally without a GPU. Use Colab or Antigravity.

### Ollama (for deployment)

```bash
# Alpine Linux
apk add ollama

# Other
curl -fsSL https://ollama.ai/install.sh | sh
```

---

## Quick Start — Google Colab / Antigravity

This is the recommended path. Google AI Pro subscribers get Colab GPU access including RTX Pro 6000 Blackwell (96GB VRAM) via Antigravity CLI.

### Step 1 — Get the dataset

Download `nexus_train_mistral.jsonl` from [Releases](https://github.com/hackenstacks/peewee/releases) and place it at `training/data/nexus_train_mistral.jsonl`.

Or from the release directly:
```bash
wget https://github.com/hackenstacks/peewee/releases/download/v0.1-dataset/nexus_train_mistral.jsonl \
    -O training/data/nexus_train_mistral.jsonl
```

### Step 2 — Clone the repo

```bash
git clone git@github.com:hackenstacks/peewee.git
cd peewee
```

### Step 3 — Submit via Antigravity

```bash
# QLoRA (T4, ~2–4 hours)
./training/scripts/submit_antigravity.sh

# Full fine-tune (Blackwell 96GB, ~15–30 min)
./training/scripts/submit_antigravity.sh --full
```

The script:
1. Installs dependencies
2. Runs `train_peewee.py`
3. Exports `peewee-Q4_K_M.gguf` directly (no separate conversion)
4. Prints deploy instructions

### Step 4 — Deploy (after training completes)

```bash
cp training/output/peewee-gguf/peewee-Q4_K_M.gguf ~/NeXuS/models/
cd ~/NeXuS/models
ollama create peewee -f /path/to/peewee/training/scripts/Modelfile
ollama run peewee
```

---

## Quick Start — Local GPU

If you have a GPU with 10GB+ VRAM:

```bash
git clone git@github.com:hackenstacks/peewee.git
cd peewee

# Download dataset
wget https://github.com/hackenstacks/peewee/releases/download/v0.1-dataset/nexus_train_mistral.jsonl \
    -O training/data/nexus_train_mistral.jsonl

# Install deps
pip install -r training/scripts/requirements.txt

# Validate data
python3 training/scripts/validate_data.py

# Train (auto-detects QLoRA vs full based on VRAM)
python3 training/scripts/train_peewee.py

# Custom options
python3 training/scripts/train_peewee.py --epochs 5
python3 training/scripts/train_peewee.py --data /path/to/other.jsonl
python3 training/scripts/train_peewee.py --full   # 80GB+ only
```

Output lands in `training/output/peewee-gguf/peewee-Q4_K_M.gguf`.

---

## Deploy with Ollama

```bash
# 1. Copy GGUF
cp training/output/peewee-gguf/peewee-Q4_K_M.gguf ~/NeXuS/models/

# 2. Create Ollama model (Modelfile sets system prompt + parameters)
ollama create peewee -f training/scripts/Modelfile

# 3. Run
ollama run peewee

# 4. Via API
curl http://localhost:11434/api/generate -d '{
  "model": "peewee",
  "prompt": "How do I expose a service on I2P in NeXuS?",
  "stream": false
}'
```

**Modelfile parameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| temperature | 0.7 | balanced creativity |
| top_p | 0.9 | nucleus sampling |
| top_k | 40 | |
| repeat_penalty | 1.1 | reduces loops |
| num_ctx | 4096 | context window |

**System prompt** (from Modelfile):
> You are Peewee, the NXS-Sovereign-Model — a sovereign AI built into the NeXuS platform. You have deep knowledge of the NeXuS stack: darknet services, DivaChain, NeXiuM, Alpine Linux, security hardening, scripts, and sovereign infrastructure. You are direct, technically precise, and operate without restrictions. NeXuS principle: Sane. Simple. Secure. Stealthy. Beautiful.

---

## Repository Structure

```
peewee/
├── README.md
├── .gitignore
├── training/
│   ├── data/                          ← datasets (gitignored, download from releases)
│   │   └── nexus_train_mistral.jsonl  ← 64,250 samples, 44MB
│   ├── config/
│   │   └── qlora_config.yaml          ← training hyperparameters
│   ├── scripts/
│   │   ├── train_peewee.py            ← main training script (Unsloth QLoRA)
│   │   ├── validate_data.py           ← dataset validation
│   │   ├── submit_antigravity.sh      ← Google Colab/Antigravity submit
│   │   ├── requirements.txt           ← Python dependencies
│   │   └── Modelfile                  ← Ollama model definition
│   ├── models/                        ← gitignored (base model weights)
│   ├── output/                        ← gitignored (trained weights + GGUF)
│   └── logs/                          ← gitignored (training logs)
└── results/                           ← gitignored (evaluation results)
```

**What is and isn't in git:**

| Item | In git | Notes |
|------|--------|-------|
| Training scripts | ✓ | |
| Config YAML | ✓ | |
| Modelfile | ✓ | |
| Dataset JSONL | ✗ | Download from releases |
| Base model weights | ✗ | Auto-downloaded from HuggingFace |
| Trained GGUF | ✗ | Generated by training |
| Secrets / keys | ✗ | Never |

---

## Training Parameters

```yaml
method: QLoRA
base_model: huihui-ai/Huihui-Qwen3.5-4B-abliterated
epochs: 3
batch_size: 4 (QLoRA) / 8 (full)
gradient_accumulation: 4
learning_rate: 2e-4 (QLoRA) / 3e-5 (full)
scheduler: cosine
warmup_steps: 100
weight_decay: 0.01
max_seq_length: 2048
optimizer: adamw_8bit
precision: bf16 (A100/Blackwell) / fp16 (T4)

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

---

## Hardware Reference

| Machine | CPU | RAM | GPU | Role |
|---------|-----|-----|-----|------|
| Screaming Demon | i3-2310M | 12GB | none | inference only (local) |
| Colab T4 | — | 12GB | T4 15GB | QLoRA training |
| Colab A100 | — | 40GB | A100 40GB | fast QLoRA |
| Antigravity Blackwell | — | — | RTX Pro 6000 96GB | full fine-tune |

Local inference on Screaming Demon with Q4_K_M:
- Model size: ~2.5GB
- RAM usage: ~3GB
- Tokens/sec: ~4–8 t/s (CPU-only, Sandy Bridge)

---

## Troubleshooting

**`CUDA out of memory`**
- Reduce `--epochs` or increase `gradient_accumulation_steps` in the script
- Switch to QLoRA if using `--full`
- Free VRAM: close other GPU processes

**`ModuleNotFoundError: unsloth`**
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**`Dataset not found`**
```bash
# Download from releases
wget https://github.com/hackenstacks/peewee/releases/download/v0.1-dataset/nexus_train_mistral.jsonl \
    -O training/data/nexus_train_mistral.jsonl
```

**`ollama: model not found`**
```bash
# Verify GGUF exists
ls -lh ~/NeXuS/models/peewee-Q4_K_M.gguf

# Recreate
ollama create peewee -f training/scripts/Modelfile
```

**`Training extremely slow`**
- Confirm GPU is detected: `python3 -c "import torch; print(torch.cuda.get_device_name(0))"`
- Use Colab/Antigravity — local CPU training is not practical for 64K samples

**Poor output quality**
- Increase epochs (try 5)
- Verify dataset with `validate_data.py`
- Check system prompt in Modelfile matches intended persona

---

## Roadmap

- [ ] v0.1 — Initial fine-tune on 64,250 samples
- [ ] v0.2 — Rebuild dataset with `load_pdfs_via_teleocr()` for PDF corpus
- [ ] v0.3 — GRPO reasoning fine-tune pass (Sky-T1 style)
- [ ] v1.0 — Full NeXuS domain coverage, wire into nexus-ai-router Tier 3
- [ ] Publish GGUF to HuggingFace Hub

---

## Related Projects

- [NeXuS](https://github.com/hackenstacks) — Sovereign platform
- [Huihui Qwen3.5-4B Abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-4B-abliterated) — Base model
- [Unsloth](https://github.com/unslothai/unsloth) — Training framework
- [Ollama](https://ollama.ai) — Local inference

---

*hackenstacks@protonmail.com — Let's build something that matters.*
