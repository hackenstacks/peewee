#!/bin/sh
# Submit Peewee training job via Antigravity CLI
# Usage: ./submit_antigravity.sh [--full]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_FILE="$REPO_DIR/training/data/nexus_train_mistral.jsonl"
TRAIN_SCRIPT="$SCRIPT_DIR/train_peewee.py"
FULL_FLAG=""

[ "$1" = "--full" ] && FULL_FLAG="--full"

echo "=== Peewee Antigravity Submit ==="
echo "Repo   : $REPO_DIR"
echo "Data   : $DATA_FILE"
echo "Mode   : ${FULL_FLAG:-QLoRA}"

if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: dataset not found at $DATA_FILE"
    echo "Download from: https://github.com/hackenstacks/peewee/releases"
    exit 1
fi

# Install deps in Colab environment
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# Run training
python3 "$TRAIN_SCRIPT" \
    --data "$DATA_FILE" \
    --output "$REPO_DIR/training/output/peewee-qlora" \
    --epochs 3 \
    $FULL_FLAG

echo ""
echo "=== Training complete ==="
echo "GGUF location: $REPO_DIR/training/output/peewee-gguf/"
echo ""
echo "To deploy locally:"
echo "  cp $REPO_DIR/training/output/peewee-gguf/peewee-Q4_K_M.gguf ~/NeXuS/models/"
echo "  cd ~/NeXuS/models && ollama create peewee -f $SCRIPT_DIR/Modelfile"
