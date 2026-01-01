import os
import json
import re
import google.generativeai as genai

IMAGE_DIR = "images"
OUTPUT_DIR = "outputs"
PROMPT_PATH = "prompt.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

genai.configure(
    api_key="AIzaSyDbvgDbE6kikBgFuQmpIitV8wfuCLl0N70"
)

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_PROMPT
)

def load_image(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def extract_json_from_gemini_response(response):
    texts = []

    for cand in response.candidates:
        if not hasattr(cand, "content"):
            continue
        for part in cand.content.parts:
            if hasattr(part, "text") and part.text:
                texts.append(part.text)

    if not texts:
        return None

    full_text = "\n".join(texts)
    cleaned = re.sub(r"```json|```", "", full_text).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("JSON parse failed. Raw text was:")
        print(cleaned)
        return None

# === 新增：數字排序 key（唯一新增的東西）===
def numeric_key(filename):
    name, _ = os.path.splitext(filename)
    return int(name)

# === 主流程 ===
for filename in sorted(os.listdir(IMAGE_DIR), key=numeric_key):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(IMAGE_DIR, filename)
    image_bytes = load_image(image_path)

    response = model.generate_content(
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": "請依照規格輸出該流程圖的 JSON"},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_bytes
                        }
                    }
                ]
            }
        ],
        generation_config={"temperature": 0}
    )

    parsed = extract_json_from_gemini_response(response)

    if parsed is None:
        print(f"[WARN] {filename}: failed to extract JSON, saved as empty object")
        parsed = {}

    output_path = os.path.join(
        OUTPUT_DIR,
        os.path.splitext(filename)[0] + ".json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    print(f"Saved {output_path}")
