import base64
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# --- PATH CONFIGURATION ---
# Use the full absolute paths you provided
BASE_PATH   = Path("/media/tm23/LENOVO/Coding/projects/stable_diffusion")
RAW_DIR     = Path("/media/tm23/LENOVO/Coding/projects/stable_diffusion/finetune_data")
DATASET_DIR = Path("/media/tm23/LENOVO/Coding/projects/stable_diffusion/finetune_data")

# Ensure the destination exists
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# --- VLLM CONFIGURATION ---
VLLM_URL = "http://localhost:8000/v1"
MODEL    = "google/gemma-4-26B-A4B-it"

PROMPT = (
    "Describe this anime image in detail for use as a Stable Diffusion prompt. "
    "Include: character gender, hair color and style, eye color, outfit/clothing, "
    "facial expression, pose, background/setting, lighting, art style, and overall mood. "
    "Output one concise natural language paragraph only. No lists, no preamble."
)

client = OpenAI(base_url=VLLM_URL, api_key="EMPTY")

def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def caption_image(img_path: Path) -> str:
    b64 = encode_image(img_path)
    # Map any common extension to the mime type the API expects
    mime_type = "jpeg" if img_path.suffix.lower() in [".jpg", ".jpeg"] else "png"
    
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{mime_type};base64,{b64}"}
                },
                {"type": "text", "text": PROMPT},
            ],
        }],
        max_tokens=300,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# --- PROCESSING ---
# Case-insensitive search for images
extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
images = []
for ext in extensions:
    images.extend(list(RAW_DIR.glob(f"**/{ext}")))

images = sorted(list(set(images))) # Remove duplicates and sort
print(f"Found {len(images)} images in {RAW_DIR}")

for img_path in tqdm(images):
    dst_img = DATASET_DIR / img_path.name
    dst_txt = DATASET_DIR / (img_path.stem + ".txt")

    # Resume-safe check
    if dst_txt.exists():
        continue 

    # Copy instead of symlink to avoid potential NTFS/FAT32 permission issues on external drives
    if not dst_img.exists():
        shutil.copy2(img_path, dst_img)

    try:
        caption = caption_image(img_path)
        dst_txt.write_text(caption)
        # Optional: Print progress updates to terminal
        # tqdm.write(f"✓ {img_path.name[:20]}...: {caption[:50]}...")
    except Exception as e:
        tqdm.write(f"✗ {img_path.name} failed: {e}")

print("Processing complete!")