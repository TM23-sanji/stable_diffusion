#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  SDXL LoRA training launcher — single A100 80GB
#  Usage:  bash scripts/train.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

echo "═══════════════════════════════════════════"
echo "  SDXL LoRA — anime fine-tune"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "═══════════════════════════════════════════"

# ── 1. Install deps with UV (Manual Path Fix) ──
echo "[1/4] Installing dependencies..."

# Install uv but ignore the shell setup errors
curl -LsSf https://astral.sh/uv/install.sh | sh || true

# Explicitly add the bin path where uv was just placed
export PATH="$HOME/.local/bin:$PATH"

# Now use uv via the direct path to be 100% safe
uv pip install --system \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

uv pip install --system \
    diffusers transformers accelerate peft safetensors \
    huggingface_hub wandb Pillow tqdm xformers bitsandbytes

# ── 2. Configure accelerate for single GPU ──
echo "[2/4] Configuring accelerate..."
accelerate config default --mixed_precision bf16

# ── 3. Sanity checks ────────────────────────
echo "[3/4] Sanity checks..."

DATA_DIR="finetune_data"
SCRIPT="scripts/train_lora_sdxl.py"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: $DATA_DIR directory not found. Make sure you cloned the repo correctly."
    exit 1
fi

NUM_IMAGES=$(find "$DATA_DIR" -name "*.jpg" | wc -l)
NUM_TEXTS=$(find  "$DATA_DIR" -name "*.txt" | wc -l)
echo "  Found $NUM_IMAGES images, $NUM_TEXTS captions"

if [ "$NUM_IMAGES" -ne "$NUM_TEXTS" ]; then
    echo "WARNING: image count ($NUM_IMAGES) != caption count ($NUM_TEXTS)"
fi

python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA!'; \
    print(f'  CUDA OK — {torch.cuda.get_device_name(0)}, \
    VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB')"

# ── 4. Launch training ───────────────────────
echo "[4/4] Starting training..."

accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    "$SCRIPT"

echo ""
echo "✓ Training complete! LoRA weights saved to output/lora/final"
echo "✓ Pushed to HuggingFace Hub as specified in train_lora_sdxl.py"
