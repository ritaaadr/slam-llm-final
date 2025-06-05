import whisper
import os
from glob import glob
import json


model = whisper.load_model("base")  # or "medium", "large", etc.

dataset_path = "dataset/training"  # or validation/test
output = []

for wav_path in sorted(glob(os.path.join(dataset_path, "*.wav"))):
    print(f"Transcribing {os.path.basename(wav_path)}...")
    result = model.transcribe(wav_path)
    text = result["text"].strip()
    
    key = os.path.basename(wav_path).replace(".wav", "_ASR")
    output.append({
        "key": key,
        "source": f"/workspace/SLAM-LLM/{wav_path}",
        "target": text
    })

# Save to JSONL
with open("dataset/jsonl/training.jsonl", "w", encoding="utf-8") as f:
    for line in output:
        f.write(json.dumps(line) + "\n")
