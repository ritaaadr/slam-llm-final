import whisper
import os
from glob import glob
import json

model = whisper.load_model("base")  # or "medium", "large", etc.

# === CONFIG ===
# Local dataset path (on host)
dataset_split = "training"  # change to "training" or "test" as needed
dataset_path = f"/home/annaritadero/slam-llm-final/dataset/{dataset_split}"

# Path to appear inside Docker container
docker_prefix = f"workspace/SLAM-LLM/dataset/{dataset_split}"

# Output JSONL
output_jsonl_path = f"/home/annaritadero/slam-llm-final/dataset/jsonl/{dataset_split}.jsonl"

# Limit samples for testing


output = []

for wav_path in sorted(glob(os.path.join(dataset_path, "*.wav"))):
    print(f"Transcribing {os.path.basename(wav_path)}...")
    result = model.transcribe(wav_path)
    text = result["text"].strip()

    key = os.path.basename(wav_path).replace(".wav", "_ASR")
    
    # Convert to Docker path
    docker_path = os.path.join(docker_prefix, os.path.basename(wav_path))
    
    output.append({
        "key": key,
        "source": docker_path,
        "target": text
    })

# Ensure output folder exists
os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

# Save to JSONL
with open(output_jsonl_path, "w", encoding="utf-8") as f:
    for line in output:
        f.write(json.dumps(line) + "\n")

print(f"\nSaved {len(output)} samples to {output_jsonl_path}")
x
