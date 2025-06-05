import os
import random
import shutil
from glob import glob
import torchaudio
from collections import defaultdict

# === CONFIG ===
LIBRISPEECH_PATH = "dataset/Librispeech/LibriSpeech/train-clean-360"

# Speakers who read selected books
ALLOWED_SPEAKER_IDS = {
    1645,7909,2146,979,7475,4381,8228,5063,6037,6727,1827,7558,7825,7657,3630,4433,6620,208,7051,8479,
    323,1885,954,2812,1933,2652,4054,8771,1290,4246,1552,2533,1734,2696,986,2815,4236,7169,815,4806,
    3792,7777,272,2654,1801,1641,1390,4807,8057,8490,3082,4734,487,5637,1547,2709,7011,3825,1634,217,
    5776,4425,6160,8875,5767,6119,8464,1779,7395,464,6054,6359,1498,835,7297,8592,249,2823,8176,7316
    #,7945,3835,549,4733,4681,3294,7733,369,2787,4846,7783,8421,1060,353,2127,7383,7384,5448,1811,6673,6258,
    #4770,7030,4860,6686,576,7498,953,7956,2592,708,5304,8887,8329,6763,8687,1079,6371,920,1012,1382,8684,
    #2827,3493,2010,7229,70,7957,8190,4138,7126,612,4335,4598,9026,7647,7145,3967,8347,8776,4837,6406,2401,
    #2229,3001,5293,6235,8050,8591,850,4519,278,7481,1224,7416,231,7949,5489,7837,830,5154,3584,3852,5684,
    #806,4257,7967,246,4289,716,5002,5242,2348,3945,6188,2393,7484,8699,1348,2056,1535,6492,5246,6060,
    #8388,1752,7881,451,157,1668,6352,6157,5007,1025,4719,6388,2598,5519,2149,1943,6378,816,3905,7555,
    #1638,7665,6937,6690,6308,6782,984,5337,3258,4044,64,4260,398,7434,7140,8404,6189,6341,7335,1031,
    #7286,1705,2882,2364,1100,5266,1383,1487,5513,8791,8825,8534,475,783,2512,6575,4238,6294,671,4133
    #,2992,8722

}

# Output folders
OUTPUT_TRAIN_DIR = "dataset/training"
OUTPUT_VAL_DIR = "dataset/validation"
OUTPUT_TEST_DIR = "dataset/test"

TRAIN_SIZE = 5000
VAL_SIZE = 550
TEST_SIZE = 550
SEED = 42
CONVERT_TO_WAV = True

def collect_speaker_files(base_path, allowed_speakers):
    files = []
    for speaker_id in allowed_speakers:
        speaker_path = os.path.join(base_path, str(speaker_id))
        if not os.path.exists(speaker_path):
            print(f"⚠️ Speaker folder {speaker_path} not found")
            continue
        for flac_file in glob(os.path.join(speaker_path, "*", "*.flac")):
            chapter_id = int(flac_file.split(os.sep)[-2])
            files.append({
                "path": flac_file,
                "speaker_id": speaker_id,
                "chapter_id": chapter_id
            })
    return files

def group_by_speaker(samples):
    grouped = defaultdict(list)
    for item in samples:
        grouped[item['speaker_id']].append(item)
    return grouped

def split_by_speaker(grouped_samples, train_n, val_n, test_n, seed=SEED):
    speakers = list(grouped_samples.keys())
    random.seed(seed)
    random.shuffle(speakers)

    train, val, test = [], [], []
    for speaker in speakers:
        samples = grouped_samples[speaker]
        if len(train) + len(samples) <= train_n:
            train.extend(samples)
        elif len(val) + len(samples) <= val_n:
            val.extend(samples)
        elif len(test) + len(samples) <= test_n:
            test.extend(samples)
        if len(train) >= train_n and len(val) >= val_n and len(test) >= test_n:
            break
    return train, val, test

def copy_and_convert(samples, target_folder, convert=CONVERT_TO_WAV):
    os.makedirs(target_folder, exist_ok=True)
    for item in samples:
        src = item["path"]
        filename = os.path.basename(src).replace(".flac", ".wav" if convert else ".flac")
        dst = os.path.join(target_folder, filename)
        if convert:
            try:
                waveform, sample_rate = torchaudio.load(src)
                torchaudio.save(dst, waveform, sample_rate)
            except Exception as e:
                print(f"Failed to convert {src}: {e}")
        else:
            shutil.copy(src, dst)

if __name__ == "__main__":
    print("Collecting files from allowed speakers...")
    metadata = collect_speaker_files(LIBRISPEECH_PATH, ALLOWED_SPEAKER_IDS)
    print(f"Found {len(metadata)} utterances from selected speakers.")

    print("Grouping by speaker...")
    grouped = group_by_speaker(metadata)

    print("Splitting dataset (speaker-independent)...")
    train, val, test = split_by_speaker(grouped, TRAIN_SIZE, VAL_SIZE, TEST_SIZE)

    print(f"\nStats:")
    print(f"Train: {len(train)} samples")
    print(f"Val:   {len(val)} samples")
    print(f"Test:  {len(test)} samples")
    print(f"Unique speakers in train: {len(set(x['speaker_id'] for x in train))}")
    print(f"Unique speakers in val:   {len(set(x['speaker_id'] for x in val))}")
    print(f"Unique speakers in test:  {len(set(x['speaker_id'] for x in test))}")

    print("\nCopying files and converting (if needed)...")
    copy_and_convert(train, OUTPUT_TRAIN_DIR)
    copy_and_convert(val, OUTPUT_VAL_DIR)
    copy_and_convert(test, OUTPUT_TEST_DIR)

    print("Done.")
