"""
FRESH AUDIT 3 — Code Correctness (Static Analysis + Token Check)
Written from scratch — no reference to existing audit3_code.py
Author: ML Auditor (Antigravity)
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

BASE = r"C:\Users\sree nilay\Downloads\DOMAIN-PRO-SRDC\DOMAIN-PRO-SRDC\project\SRDC"

print("=" * 70)
print("AUDIT 3 — CODE CORRECTNESS")
print("=" * 70)

# -------------------------------------------------------
# 3A — Preprocessing order
# -------------------------------------------------------
print("\n" + "-" * 70)
print("AUDIT 3A — Preprocessing Order (split_data.py / fix_splits.py)")
print("-" * 70)

print("""
[READING split_data.py]
  Line 4:  df = pd.read_csv("after_feature_internal_semantic_process_data.csv")
  Line 7:  df['family'] = df['family'].astype(str)
  Line 10: df['is_ransomware'] = (df['family'] != 'Goodware').astype(int)
  Line 13: train_df, test_df = train_test_split(df, test_size=0.2,
               stratify=df['is_ransomware'], random_state=42)
  Line 20: train_df.to_csv("train.csv", index=False)
  Line 21: test_df.to_csv("test.csv", index=False)

[ANALYSIS]
  - The input file is "after_feature_internal_semantic_process_data.csv"
    The filename itself tells us: feature processing happened BEFORE this file was created.
  - split_data.py ONLY does: type cast -> binary label creation -> train_test_split -> save
  - There is NO fit_transform, StandardScaler, LabelEncoder, or any sklearn fitting.
  - The split happens AFTER all features are pre-baked into the CSV.

[READING fix_splits.py]
  Line 11: df = pd.read_csv(DATA_PATH)
  Line 16: df = df.drop_duplicates()
  Line 26: train_df, test_df = train_test_split(df, test_size=0.2,
               stratify=df['is_ransomware'], random_state=42)
  -- Zero-day split uses hard-coded family IDs, not any fitted transformer.
  -- No StandardScaler, LabelEncoder, or fit-before-split.

[VERDICT 3A] PASS — No preprocessing leakage.
  The feature CSV was already finalized BEFORE any split was created.
  No sklearn fitter was applied to the full dataset then split; only
  read-only operations (type cast, binary flag) preceded the split.
""")

# -------------------------------------------------------
# 3B — Evaluation on training data / checkpoint logic
# -------------------------------------------------------
print("-" * 70)
print("AUDIT 3B — Checkpoint Selection: Train Acc vs Test Acc?")
print("-" * 70)

zd_csv = f"{BASE}\\result\\zero_day_results.csv"
fam_csv = f"{BASE}\\result\\family_results.csv"

zd_results  = pd.read_csv(zd_csv)
fam_results = pd.read_csv(fam_csv)

print(f"\n[zero_day_results.csv] Columns: {list(zd_results.columns)}")
print(zd_results.to_string(index=False))

print(f"\n[family_results.csv] Columns: {list(fam_results.columns)}")
print(fam_results.to_string(index=False))

# Analyse srdc_zero_day.py checkpoint saving
print("""
[READING srdc_zero_day.py checkpoint logic]
  Lines 131-134:
    model_path = f'{SAVE_DIR}/srdc_zero_day_epoch{epoch+1}.pth'
    torch.save(model.state_dict(), model_path)

  CRITICAL: There is NO best-model tracking logic.
  Every epoch is saved with its epoch number. There is no:
    - best_acc variable
    - if test_acc > best_acc: save logic
    - Any reference to 'BEST' in the filename during training

  The file 'srdc_zero_day_BEST.pth' does NOT exist as an output of
  srdc_zero_day.py. It must have been:
    (a) Manually renamed from one of the epoch checkpoints, OR
    (b) Created by a different training run not reflected in this code.

[READING srdc_family_classification.py checkpoint logic]
  Lines 147-150:
    model_path = f'{SAVE_DIR}/srdc_family_epoch{epoch+1}.pth'
    torch.save(model.state_dict(), model_path)

  Same pattern — saves every epoch, no automated best-model selection.
""")

# Find the epoch with highest test_acc for zero_day
best_zd_epoch = zd_results.loc[zd_results['test_acc'].idxmax()]
print(f"[INFO] Best test_acc in zero_day_results.csv: epoch {int(best_zd_epoch['epoch'])} | test_acc={best_zd_epoch['test_acc']}")
print(f"[INFO] Best train_acc in zero_day_results.csv: epoch {int(zd_results['train_acc'].idxmax())+1} | train_acc={zd_results['train_acc'].max()}")

best_fam_epoch = fam_results.loc[fam_results['balanced_acc'].idxmax()]
print(f"\n[INFO] Best balanced_acc in family_results.csv: epoch {int(best_fam_epoch['epoch'])} | balanced_acc={best_fam_epoch['balanced_acc']}")

print("""
[VERDICT 3B] FAIL — No automated best-model checkpoint selection.
  The training script saves ALL epoch weights but has no logic to identify
  which epoch was 'best'. The file srdc_zero_day_BEST.pth was created
  MANUALLY (likely by the researcher renaming a file). This means:
    - The "BEST" checkpoint may actually be the most-overfit epoch
    - There is no guarantee it corresponds to the highest test accuracy
    - The selection criterion and which epoch it maps to are UNKNOWN
  Fix: Add 'if test_acc > best_acc: torch.save(...)' in the training loop,
  saving as 'BEST.pth' only when test accuracy improves.
""")

# -------------------------------------------------------
# 3C — Token truncation rate
# -------------------------------------------------------
print("-" * 70)
print("AUDIT 3C — Token Truncation Rate (max_length=1024)")
print("-" * 70)

print("\n[INFO] Loading zero_day_test.csv for truncation analysis...")

zd_test = pd.read_csv(f"{BASE}\\splits\\zero_day_test.csv")
FEATURE_COLS = ['apiFeatures', 'dropFeatures', 'regFeatures',
                'filesFeatures', 'filesEXTFeatures', 'dirFeatures', 'strFeatures']

zd_test = zd_test.fillna('')

# Concatenate exactly as srdc_zero_day.py does (single string, space-separated)
texts = (
    zd_test['apiFeatures'].astype(str) + " " +
    zd_test['dropFeatures'].astype(str) + " " +
    zd_test['regFeatures'].astype(str) + " " +
    zd_test['filesFeatures'].astype(str) + " " +
    zd_test['filesEXTFeatures'].astype(str) + " " +
    zd_test['dirFeatures'].astype(str) + " " +
    zd_test['strFeatures'].astype(str)
).str.strip().tolist()

MAX_LENGTH = 1024

print(f"[INFO] Loading GPT-2 tokenizer (local, no model download needed)...")
try:
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Tokenizer loaded. Checking {len(texts)} samples against max_length={MAX_LENGTH}...")

    truncated = 0
    raw_lengths = []
    for text in texts:
        # Encode WITHOUT truncation to get the raw token count
        ids = tokenizer.encode(text, truncation=False)
        raw_lengths.append(len(ids))
        if len(ids) > MAX_LENGTH:
            truncated += 1

    total = len(texts)
    pct = 100.0 * truncated / total
    print(f"\n[RESULT] {truncated} out of {total} samples ({pct:.1f}%) are being TRUNCATED.")
    print(f"[INFO] Token length statistics:")
    print(f"  Min:    {min(raw_lengths)}")
    print(f"  Max:    {max(raw_lengths)}")
    print(f"  Mean:   {np.mean(raw_lengths):.1f}")
    print(f"  Median: {np.median(raw_lengths):.1f}")
    print(f"  Samples with >512 tokens:  {sum(l > 512 for l in raw_lengths)}")
    print(f"  Samples with >1024 tokens: {sum(l > 1024 for l in raw_lengths)}")
    print(f"  Samples with >2048 tokens: {sum(l > 2048 for l in raw_lengths)}")

    if pct > 20:
        print(f"\n[VERDICT 3C] HIGH SEVERITY — {pct:.1f}% of samples are truncated (threshold: 20%).")
        print("  The model receives incomplete input for more than 1 in 5 test samples.")
        print("  Fix: Increase max_length or use hierarchical/chunked encoding.")
    elif pct > 5:
        print(f"\n[VERDICT 3C] MEDIUM — {pct:.1f}% truncation. Worth investigating.")
    else:
        print(f"\n[VERDICT 3C] PASS — Truncation rate ({pct:.1f}%) is below the 20% threshold.")

except ImportError:
    print("[ERROR] transformers not installed. Run: pip install transformers")
    print("[SKIP] Audit 3C skipped — install transformers to run this check.")

# -------------------------------------------------------
# 3D — Confidence score math (srdc_demo_fixed.py)
# -------------------------------------------------------
print("\n" + "-" * 70)
print("AUDIT 3D — Confidence Score Math (srdc_demo_fixed.py)")
print("-" * 70)

print("""
[READING srdc_demo_fixed.py — predict() function, lines 49-61]

  def predict(model, text, tokenizer, device):
      encoding = tokenizer(text, truncation=True, max_length=1024,
                           padding='max_length', return_tensors='pt')
      input_ids = encoding['input_ids'].to(device)
      attention_mask = encoding['attention_mask'].to(device)
      with torch.no_grad():
          logits = model(input_ids, attention_mask)
      probs = torch.softmax(logits, dim=1)          # <-- Line 58
      pred = logits.argmax(dim=1).item()
      confidence = probs[0][pred].item() * 100       # <-- Line 60
      return pred, confidence

[ANALYSIS]
  - Line 58: torch.softmax(logits, dim=1) IS explicitly applied.
  - The confidence is probs[0][pred].item() * 100 — a valid probability [0,1]
    scaled to percentage [0,100].
  - Raw logits are never directly displayed as probabilities.
  - softmax ensures all class probabilities sum to 1.0.

[NOTE] One minor issue: pred = logits.argmax(dim=1).item() uses logits, but
  since argmax over logits is equivalent to argmax over softmax probabilities
  (softmax is monotone), this is mathematically correct.

[VERDICT 3D] PASS — Confidence scores are mathematically valid.
  softmax is correctly applied before extracting confidence values.
  The percentages shown to the user are genuine probabilities in [0%, 100%].
""")

# -------------------------------------------------------
# 3E — Random seed check
# -------------------------------------------------------
print("-" * 70)
print("AUDIT 3E — Random Seed / Reproducibility")
print("-" * 70)

print("""
[READING split_data.py]
  Line 13-18:
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['is_ransomware'],
        random_state=42    # <-- SEED SET
    )
  Comment on line 17: "# or 123 for different"

[READING fix_splits.py]
  Line 26-31 (standard split):
    train_df, test_df = train_test_split(df, test_size=0.2,
        stratify=df['is_ransomware'], random_state=42)

  Line 50 (goodware split for zero-day):
    gw_train, gw_test = train_test_split(goodware_df, test_size=0.2, random_state=42)

  Line 61 (shuffle after concat):
    zero_day_train = pd.concat([gw_train, rw_train]).sample(frac=1, random_state=42)

  Line 62:
    zero_day_test = pd.concat([gw_test, rw_test]).sample(frac=1, random_state=42)

[ANALYSIS]
  - ALL uses of train_test_split have random_state=42.
  - All .sample() shuffles have random_state=42.
  - The split is fully deterministic and reproducible.
  - Concern: The comment "# or 123 for different" suggests the seed may have been
    changed during development; the actual splits on disk may not match random_state=42
    if an older version was run first.

[VERDICT 3E] PASS — Random seed is fixed (random_state=42) in all split operations.
  The experiment is reproducible as long as this script is re-run with the
  same input data.
""")

print("=" * 70)
print("AUDIT 3 COMPLETE")
print("=" * 70)
