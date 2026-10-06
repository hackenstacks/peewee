# Peewee Training on NeXuS Cannon

**Project:** Train Mistral Vibe (Peewee) using NeXuS Cannon infrastructure
**Status:** Planning Phase
**Location:** /home/user/mistral/peewee/
**Last Updated:** October 5, 2026

---

## OVERVIEW

This directory contains the configuration and scripts for training Peewee (Mistral Vibe) on the NeXuS Cannon infrastructure using Mistrals online training service.

### Why Not Local?
- Local training on Screaming Demon would take ~9 days
- Using Mistrals online service provides faster training with cloud resources
- NeXuS Cannon provides the training framework and infrastructure

---

## PROJECT STRUCTURE

\`\`\`
training/
  data/         - Training datasets
  models/       - Model files
  config/       - Training configurations
  output/       - Training outputs
  logs/         - Training logs
  scripts/      - Training scripts
results/        - Training results
\`\`\`

---

## QUICK START

1. Organize files:
\`\`\`bash
cd /home/user/mistral/peewee
mkdir -p training/{data,models,output,logs,scripts} results
\`\`\`

2. Add your training datasets to training/data/

3. Add Peewee model to training/models/peewee-base/

4. Start training:
\`\`\`bash
./training/scripts/train_qlora.sh
\`\`\`

---

## TRAINING METHODS

- **QLoRA** - Quantized Low-Rank Adaptation (Recommended - Fast, Low Memory)
- **LoRA**  - Low-Rank Adaptation (Medium Speed, Medium Memory)
- **GRPO**  - Gradient-based Reinforcement Learning (Slow, High Memory)

---

## TRAINING WORKFLOW

1. Prepare Environment: \`mkdir -p training/{data,models,output,logs,scripts}\`
2. Add Your Data: \`cp /path/to/dataset.jsonl training/data/\`
3. Add Peewee Model: \`cp -r /path/to/model/* training/models/peewee-base/\`
4. Configure Training: Edit files in training/config/
5. Prepare Data: \`python training/scripts/prepare_data.py\`
6. Start Training: \`./training/scripts/train_qlora.sh\`
7. Monitor Training: \`tail -f training/logs/train_qlora_*.log\`
8. Evaluate Results: \`python training/scripts/evaluate.py output/ test.jsonl\`

---

## NEXUS CANNON INTEGRATION

Access via:
1. Neural Nexus Web: https://newneuralnexus.com/
2. CLI: nexus-cannon train --help
3. API: POST to https://api.neuralnexus.com/train

---

## DATASET PREPARATION

Supported Formats:
- JSONL (Recommended): {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
- JSON: Array of conversation objects
- Text: Simple user/assistant format

---

## TRAINING PARAMETERS

| Method | Epochs | Batch Size | Learning Rate | Max Length |
|--------|--------|------------|---------------|------------|
| QLoRA  | 3-5    | 4-8        | 2e-5 to 5e-5  | 2048       |
| LoRA   | 5-10   | 8-16       | 3e-5 to 5e-5  | 2048       |
| GRPO   | 10-20  | 4-8        | 1e-5 to 2e-5  | 1024       |

---

## TROUBLESHOOTING

- Data file not found: Run python training/scripts/prepare_data.py
- Model directory not found: Copy model to training/models/peewee-base/
- Out of memory: Reduce batch size or use QLoRA
- Training too slow: Use Mistrals online service
- Poor results: More data, more epochs, check quality

---

## NEXT STEPS

1. Set up directory structure
2. Add training datasets to training/data/
3. Add Peewee model to training/models/peewee-base/
4. Configure training in training/config/
5. Start training via NeXuS Cannon

---

Status: Ready for data and model files
Author: Mistral Vibe (Peewee)
For: NeXuS Project - Peewee Training Initiative
Location: /home/user/mistral/peewee/README.md

Lets build something that matters. - NeXuS Principle
