"""
SDXL LoRA fine-tuning with diffusers.
- Reads finetune_data/{N}.jpg + {N}.txt pairs
- Bucket-aware (AspectRatioBucket via custom sampler)
- WandB tracking
- Pushes final LoRA to HuggingFace Hub on completion
"""

import os, math, random, logging, itertools
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from tqdm.auto import tqdm

import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr

from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from peft import LoraConfig, get_peft_model
from huggingface_hub import HfApi

# ─────────────────────────────────────────────
# HARDCODED CONFIG  (pod is ephemeral, no .env)
# ─────────────────────────────────────────────

HF_TOKEN           = ""
WANDB_API_KEY      = ""
HF_REPO_ID         = "tm23hgf/anime-sdxl-lora"   # will be created if not exists

BASE_MODEL         = "stabilityai/stable-diffusion-xl-base-1.0"
DATA_DIR           = Path("finetune_data")
OUTPUT_DIR         = Path("output/lora")

# Training hyperparams
MAX_TRAIN_STEPS    = 2000
LEARNING_RATE      = 1e-4
LR_WARMUP_STEPS    = 200
TRAIN_BATCH_SIZE   = 1
GRAD_ACCUM_STEPS   = 4
MIXED_PRECISION    = "bf16"
SEED               = 42

# LoRA
LORA_RANK          = 32
LORA_ALPHA         = 16
LORA_DROPOUT       = 0.05

# Bucketing — resolutions to group images into
BUCKET_RESOLUTIONS = [
    (1024, 1024),
    (1152, 896), (896, 1152),
    (1216, 832), (832, 1216),
    (1344, 768), (768, 1344),
]
MAX_TOKEN_LENGTH   = 77
CHECKPOINTING_STEPS = 500

# ─────────────────────────────────────────────

logger = get_logger(__name__, log_level="INFO")


# ── Dataset ─────────────────────────────────

def find_bucket(w: int, h: int) -> tuple[int, int]:
    """Return the bucket (bw, bh) with the closest aspect ratio."""
    img_ar = w / h
    return min(BUCKET_RESOLUTIONS, key=lambda b: abs((b[0] / b[1]) - img_ar))


class AnimeLoRADataset(Dataset):
    def __init__(self, data_dir: Path, tokenizer1, tokenizer2):
        self.pairs = sorted(
            [(p, p.with_suffix(".txt"))
             for p in data_dir.glob("*.jpg")
             if p.with_suffix(".txt").exists()]
        )
        assert len(self.pairs) > 0, f"No image/txt pairs found in {data_dir}"
        logger.info(f"Dataset: {len(self.pairs)} image-caption pairs")

        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2

    def __len__(self):
        return len(self.pairs)

    def tokenize(self, caption: str):
        def _tok(tokenizer):
            ids = tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=MAX_TOKEN_LENGTH,
                return_tensors="pt",
            ).input_ids
            return ids[0]
        return _tok(self.tokenizer1), _tok(self.tokenizer2)

    def __getitem__(self, idx):
        img_path, txt_path = self.pairs[idx]
        caption = txt_path.read_text(encoding="utf-8").strip()

        img = Image.open(img_path).convert("RGB")
        bw, bh = find_bucket(*img.size)
        img = img.resize((bw, bh), Image.LANCZOS)

        # to tensor [-1, 1]
        pixel = torch.tensor(
            list(img.getdata()), dtype=torch.float32
        ).reshape(bh, bw, 3).permute(2, 0, 1) / 127.5 - 1.0

        tok1, tok2 = self.tokenize(caption)
        return {"pixel_values": pixel, "input_ids_1": tok1, "input_ids_2": tok2}


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids_1":  torch.stack([b["input_ids_1"]  for b in batch]),
        "input_ids_2":  torch.stack([b["input_ids_2"]  for b in batch]),
    }


# ── Text encoding helpers ────────────────────

def encode_prompt(text_encoders, tokenizer_ids_list):
    """Run both SDXL text encoders, return (prompt_embeds, pooled_embeds)."""
    te1, te2 = text_encoders
    ids1, ids2 = tokenizer_ids_list

    with torch.no_grad():
        out1 = te1(ids1, output_hidden_states=True)
        out2 = te2(ids2, output_hidden_states=True)

    # SDXL uses second-to-last hidden state from encoder 1
    # and last hidden state + pooled from encoder 2
    hidden1  = out1.hidden_states[-2]                   # (B, 77, 768)
    hidden2  = out2.hidden_states[-2]                   # (B, 77, 1280)
    pooled   = out2[0]                                  # (B, 1280)
    embeds   = torch.cat([hidden1, hidden2], dim=-1)    # (B, 77, 2048)
    return embeds, pooled


# ── Main ────────────────────────────────────

def main():
    # Init WandB
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    os.environ["HF_TOKEN"]      = HF_TOKEN

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    accelerator_project_cfg = ProjectConfiguration(
        project_dir=str(OUTPUT_DIR),
        logging_dir=str(OUTPUT_DIR / "logs"),
    )

    accelerator = Accelerator(
        mixed_precision=MIXED_PRECISION,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        log_with="wandb",
        project_config=accelerator_project_cfg,
    )

    if accelerator.is_main_process:
        wandb.login(key=WANDB_API_KEY)
        accelerator.init_trackers(
            project_name="anime-sdxl-lora",
            config={
                "base_model":        BASE_MODEL,
                "max_train_steps":   MAX_TRAIN_STEPS,
                "learning_rate":     LEARNING_RATE,
                "lora_rank":         LORA_RANK,
                "lora_alpha":        LORA_ALPHA,
                "batch_size":        TRAIN_BATCH_SIZE,
                "grad_accum":        GRAD_ACCUM_STEPS,
                "mixed_precision":   MIXED_PRECISION,
            },
        )

    set_seed(SEED)
    logging.basicConfig(level=logging.INFO)

    # ── Load models (frozen) ────────────────
    logger.info("Loading SDXL components...")

    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

    tokenizer1 = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    tokenizer2 = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer_2")

    text_encoder1 = CLIPTextModel.from_pretrained(
        BASE_MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    text_encoder2 = CLIPTextModelWithProjection.from_pretrained(
        BASE_MODEL, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
    )

    vae = AutoencoderKL.from_pretrained(
        BASE_MODEL, subfolder="vae", torch_dtype=torch.bfloat16
    )

    unet = UNet2DConditionModel.from_pretrained(
        BASE_MODEL, subfolder="unet", torch_dtype=torch.bfloat16
    )

    # Freeze everything except UNet (which gets LoRA)
    vae.requires_grad_(False)
    text_encoder1.requires_grad_(False)
    text_encoder2.requires_grad_(False)
    unet.requires_grad_(False)

    # ── Attach LoRA to UNet ─────────────────
    logger.info(f"Attaching LoRA rank={LORA_RANK} alpha={LORA_ALPHA}")

    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
        ],
        bias="none",
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.print_trainable_parameters()

    unet.enable_gradient_checkpointing()

    # ── Dataset & DataLoader ────────────────
    dataset = AnimeLoRADataset(DATA_DIR, tokenizer1, tokenizer2)
    dataloader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # ── Optimizer ───────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # Steps per epoch for scheduler
    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / GRAD_ACCUM_STEPS
    )
    num_train_epochs = math.ceil(MAX_TRAIN_STEPS / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=LR_WARMUP_STEPS * GRAD_ACCUM_STEPS,
        num_training_steps=MAX_TRAIN_STEPS * GRAD_ACCUM_STEPS,
    )

    # ── Accelerate prepare ──────────────────
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    text_encoder1 = text_encoder1.to(accelerator.device, dtype=torch.bfloat16)
    text_encoder2 = text_encoder2.to(accelerator.device, dtype=torch.bfloat16)
    vae           = vae.to(accelerator.device, dtype=torch.bfloat16)

    # ── Training loop ───────────────────────
    logger.info(f"Starting training: {MAX_TRAIN_STEPS} steps")
    global_step = 0
    progress    = tqdm(total=MAX_TRAIN_STEPS, desc="Training", disable=not accelerator.is_main_process)

    unet.train()

    for epoch in range(num_train_epochs):
        for batch in dataloader:
            with accelerator.accumulate(unet):

                # Encode images → latents
                with torch.no_grad():
                    latents = vae.encode(
                        batch["pixel_values"].to(dtype=torch.bfloat16)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise + timesteps
                noise     = torch.randn_like(latents)
                bsz       = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encode text
                prompt_embeds, pooled_embeds = encode_prompt(
                    (text_encoder1, text_encoder2),
                    (batch["input_ids_1"], batch["input_ids_2"]),
                )

                # SDXL additional conditioning (time ids)
                # original_size, crops_coords_top_left, target_size
                add_time_ids = torch.tensor(
                    [[1024, 1024, 0, 0, 1024, 1024]] * bsz,
                    device=latents.device, dtype=torch.bfloat16,
                )

                added_cond_kwargs = {
                    "text_embeds": pooled_embeds.to(dtype=torch.bfloat16),
                    "time_ids":    add_time_ids,
                }

                # Forward
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds.to(dtype=torch.bfloat16),
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                # Loss (epsilon prediction)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ── Logging & checkpointing ──────
            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)

                if accelerator.is_main_process:
                    accelerator.log(
                        {
                            "train/loss": loss.detach().item(),
                            "train/lr":   lr_scheduler.get_last_lr()[0],
                            "train/step": global_step,
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

                if global_step % CHECKPOINTING_STEPS == 0 and accelerator.is_main_process:
                    ckpt_dir = OUTPUT_DIR / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(unet)
                    unwrapped.save_pretrained(str(ckpt_dir))
                    logger.info(f"Checkpoint saved → {ckpt_dir}")

            if global_step >= MAX_TRAIN_STEPS:
                break

        if global_step >= MAX_TRAIN_STEPS:
            break

    # ── Save & push to Hub ───────────────────
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info("Training complete. Saving final LoRA weights...")

        final_dir = OUTPUT_DIR / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        unwrapped = accelerator.unwrap_model(unet)
        unwrapped.save_pretrained(str(final_dir))

        logger.info(f"Pushing LoRA to HuggingFace Hub → {HF_REPO_ID}")
        api = HfApi(token=HF_TOKEN)
        api.create_repo(HF_REPO_ID, exist_ok=True, private=True)
        api.upload_folder(
            folder_path=str(final_dir),
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message=f"LoRA weights — {MAX_TRAIN_STEPS} steps, rank {LORA_RANK}",
        )
        logger.info(f"✓ Uploaded to https://huggingface.co/{HF_REPO_ID}")

        accelerator.end_training()

    logger.info("Done.")


if __name__ == "__main__":
    main()
