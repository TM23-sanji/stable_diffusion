import base64
import shutil
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# --- PATH CONFIGURATION ---
BASE_PATH   = Path("/media/tm23/LENOVO/Coding/projects/stable_diffusion")
# Assuming your source images are in 'data/raw'
RAW_DIR     = BASE_PATH / "finetune_data" 
# Where the images + long captions + short prompts go
DATASET_DIR = BASE_PATH / "finetune_data"

DATASET_DIR.mkdir(parents=True, exist_ok=True)

# --- VLLM CONFIGURATION ---
VLLM_URL = "http://localhost:8000/v1"
MODEL    = "google/gemma-4-26B-A4B-it"

client = OpenAI(base_url=VLLM_URL, api_key="EMPTY")

# --- PROMPT TEMPLATES ---
DESC_PROMPT = (
    "Describe this anime image in detail for a Stable Diffusion prompt. "
    "Include: character gender, hair color/style, eyes, outfit, expression, pose, "
    "background, lighting, art style, and mood. One concise paragraph. No preamble."
)

COMPRESS_PROMPT = (
    "Rewrite the following image description as a Stable Diffusion prompt "
    "under 60 words. Keep: art style, character appearance, setting, mood, lighting. "
    "Remove filler words. Output only the prompt, no preamble.\n\n"
    "Description: {caption}"
)

def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def get_vlm_response(messages: list) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=300,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# --- PROCESSING ---
extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
images = []
for ext in extensions:
    images.extend(list(RAW_DIR.glob(f"**/{ext}")))

images = sorted(list(set(images)))
print(f"Found {len(images)} images. Target: {DATASET_DIR}")

for img_path in tqdm(images):
    dst_img    = DATASET_DIR / img_path.name
    dst_txt    = DATASET_DIR / (img_path.stem + ".txt")      # Long version
    dst_prompt = DATASET_DIR / (img_path.stem + ".prompt")   # Compressed version

    # 1. Handle File Copy
    if not dst_img.exists():
        shutil.copy2(img_path, dst_img)

    try:
        # 2. Generate Detailed Caption (Image Input)
        if not dst_txt.exists():
            b64 = encode_image(img_path)
            mime = "jpeg" if img_path.suffix.lower() in [".jpg", ".jpeg"] else "png"
            
            caption = get_vlm_response([{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                    {"type": "text", "text": DESC_PROMPT},
                ],
            }])
            dst_txt.write_text(caption)
        else:
            caption = dst_txt.read_text()

        # 3. Generate Compressed Prompt (Text-only Input)
        if not dst_prompt.exists():
            short_prompt = get_vlm_response([{
                "role": "user",
                "content": [{"type": "text", "text": COMPRESS_PROMPT.format(caption=caption)}],
            }])
            dst_prompt.write_text(short_prompt)

    except Exception as e:
        tqdm.write(f"✗ {img_path.name} failed: {e}")

print("Done! Check 'finetune_data' for .txt and .prompt files.")