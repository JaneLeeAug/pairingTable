import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import random
import re

# ====== config ======
CSV_PATH = "pairing_table.csv"
OUT_DIR = Path("pairing_table_analysis")
CHUNKSIZE = 500_000

SPEAKER_COL = "speaker_id"
CONTENT_COL = "content_to_synthesize"

# 若你想指定 speaker_id，填字串；若為 None，程式會隨機抽一個 speaker_id
TARGET_SPEAKER_ID = None
# ====================

OUT_DIR.mkdir(parents=True, exist_ok=True)


def sentence_len(text):
    """
    中文一字算一 token，英文單字算一 token。
    這跟你之前 sentence_length 的算法相近。
    """
    if pd.isna(text):
        return 0
    text = str(text)
    return len(re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]", text))


# ---------------------------------------------------------
# Pass 1: 統計每個 speaker_id 幾列，並收集 speaker_id
# ---------------------------------------------------------

speaker_counts = Counter()

print("Pass 1: counting rows per speaker_id...")

for chunk in pd.read_csv(
    CSV_PATH,
    usecols=[SPEAKER_COL],
    chunksize=CHUNKSIZE,
    dtype={SPEAKER_COL: "string"},
):
    speaker_counts.update(chunk[SPEAKER_COL].dropna().tolist())

if not speaker_counts:
    raise ValueError("No speaker_id found.")

if TARGET_SPEAKER_ID is None:
    TARGET_SPEAKER_ID = random.choice(list(speaker_counts.keys()))

print(f"Selected speaker_id: {TARGET_SPEAKER_ID}")
print(f"Rows for selected speaker: {speaker_counts[TARGET_SPEAKER_ID]}")


# ---------------------------------------------------------
# Plot 2: 每個 unique speaker_id 有幾列的分佈圖
# ---------------------------------------------------------

speaker_row_counts = list(speaker_counts.values())

plt.figure(figsize=(10, 6))
plt.hist(speaker_row_counts, bins=100)
plt.xlabel("Rows per speaker_id")
plt.ylabel("Number of speakers")
plt.title("Distribution of Row Counts per speaker_id")
plt.tight_layout()
plt.savefig(OUT_DIR / "speaker_id_row_count_distribution.png", dpi=200)
plt.close()

print("Saved:", OUT_DIR / "speaker_id_row_count_distribution.png")


# ---------------------------------------------------------
# Pass 2:
# 1. 找該 speaker_id 的所有 content_to_synthesize 句長
# 3. 統計每個 unique content_to_synthesize 出現幾列
# ---------------------------------------------------------

selected_speaker_sentence_lengths = []
content_counts = Counter()

print("Pass 2: counting content_to_synthesize and selected speaker sentence lengths...")

for chunk in pd.read_csv(
    CSV_PATH,
    usecols=[SPEAKER_COL, CONTENT_COL],
    chunksize=CHUNKSIZE,
    dtype={SPEAKER_COL: "string", CONTENT_COL: "string"},
):
    # 統計所有 content_to_synthesize 出現次數
    content_counts.update(chunk[CONTENT_COL].dropna().tolist())

    # 抽出指定 speaker_id 的所有列
    selected = chunk.loc[chunk[SPEAKER_COL] == TARGET_SPEAKER_ID, CONTENT_COL]
    selected_speaker_sentence_lengths.extend(selected.map(sentence_len).tolist())


# ---------------------------------------------------------
# Plot 1: 某個 speaker_id 的 content_to_synthesize 句長分佈
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.hist(selected_speaker_sentence_lengths, bins=80)
plt.xlabel("Sentence length")
plt.ylabel("Number of rows")
plt.title(f"Sentence Length Distribution for speaker_id={TARGET_SPEAKER_ID}")
plt.tight_layout()
plt.savefig(OUT_DIR / f"sentence_length_distribution_speaker_{TARGET_SPEAKER_ID}.png", dpi=200)
plt.close()

print("Saved:", OUT_DIR / f"sentence_length_distribution_speaker_{TARGET_SPEAKER_ID}.png")


# ---------------------------------------------------------
# Plot 3: 每個 unique content_to_synthesize 出現列數的分佈
# ---------------------------------------------------------

content_row_counts = list(content_counts.values())

plt.figure(figsize=(10, 6))
plt.hist(content_row_counts, bins=100)
plt.xlabel("Rows per unique content_to_synthesize")
plt.ylabel("Number of unique contents")
plt.title("Distribution of Row Counts per unique content_to_synthesize")
plt.tight_layout()
plt.savefig(OUT_DIR / "content_to_synthesize_row_count_distribution.png", dpi=200)
plt.close()

print("Saved:", OUT_DIR / "content_to_synthesize_row_count_distribution.png")


# ---------------------------------------------------------
# Optional: 存 summary CSV，方便檢查
# ---------------------------------------------------------

pd.DataFrame({
    "speaker_id": list(speaker_counts.keys()),
    "row_count": list(speaker_counts.values()),
}).sort_values("row_count", ascending=False).to_csv(
    OUT_DIR / "speaker_id_row_counts.csv",
    index=False,
)

pd.DataFrame({
    "content_to_synthesize": list(content_counts.keys()),
    "row_count": list(content_counts.values()),
}).sort_values("row_count", ascending=False).to_csv(
    OUT_DIR / "content_to_synthesize_row_counts.csv",
    index=False,
)

print("Done.")
print("Output directory:", OUT_DIR)






"""
import pandas as pd
import random
from collections import Counter

CSV_PATH = "pairing_table.csv"
CHUNKSIZE = 500_000
SPEAKER_COL = "speaker_id"

speaker_counts = Counter()

print("Counting rows per speaker_id...")

# 讀 chunk 累積計數
for chunk in pd.read_csv(
    CSV_PATH,
    usecols=[SPEAKER_COL],
    chunksize=CHUNKSIZE,
    dtype={SPEAKER_COL: "string"},
):
    speaker_counts.update(chunk[SPEAKER_COL].dropna().tolist())

print(f"Total unique speaker_id: {len(speaker_counts)}")

# 隨機抽 100 個（如果不到 100 就全取）
sample_size = min(100, len(speaker_counts))
sampled_speakers = random.sample(list(speaker_counts.keys()), sample_size)

print("\nSampled speaker_id row counts:\n")

for spk in sampled_speakers:
    print(f"{spk}: {speaker_counts[spk]}")"""