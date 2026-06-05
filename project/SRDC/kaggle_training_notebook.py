# =============================================================================
# SRDC MaxPooling Training — Kaggle Notebook
# =============================================================================
# Instructions:
#   1. Create a new Kaggle Notebook
#   2. Upload the following files as a Kaggle Dataset:
#      - clean_splits/zero_day_train.csv
#      - clean_splits/zero_day_test.csv
#      - clean_splits/train.csv
#      - clean_splits/test.csv
#      - srdc_maxpooling_training.py
#   3. In Kaggle Notebook Settings → Accelerator → Select "GPU T4 x2"
#   4. Copy-paste this entire file into the notebook and run all cells.
#   5. After training completes, download the _BEST.pth files from
#      /kaggle/working/maxpooling_results/
# =============================================================================

# %% [markdown]
# # 🛡️ SRDC Shield — MaxPooling Combinatorial Model Training
#
# This notebook trains the **corrected** hierarchical MaxPooling architecture
# from the SRDC research paper. It fixes:
# 1. **Pooling dimension bug** (transpose before pooling)
# 2. **Data leakage** (using clean, deduplicated splits)
# 3. **Token truncation** (each feature is tokenized independently)
# 4. **Checkpoint selection** (automated best-accuracy saving)

# %% [markdown]
# ## Step 0: Verify GPU

# %%
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected! Training will be extremely slow.")
    print("Please enable GPU in Kaggle Settings → Accelerator → GPU T4 x2")

# %% [markdown]
# ## Step 1: Setup & Install Dependencies

# %%
# Install required packages (should already be available on Kaggle)
import subprocess
subprocess.run(["pip", "install", "-q", "transformers", "tqdm", "scikit-learn", "pandas"], check=True)

# %% [markdown]
# ## Step 2: Configure Data Paths
#
# **IMPORTANT**: Update `DATA_DIR` below to match where you uploaded your CSV files.
# If you uploaded them as a Kaggle Dataset named "srdc-clean-splits", the path
# will typically be `/kaggle/input/srdc-clean-splits/`.

# %%
import os

# === UPDATE THIS PATH to match your Kaggle dataset location ===
DATA_DIR = "/kaggle/input/srdc-clean-splits"

# Verify files exist
required_files = [
    "zero_day_train.csv", "zero_day_test.csv",
    "train.csv", "test.csv"
]

print("Checking for required data files...")
for f in required_files:
    path = os.path.join(DATA_DIR, f)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✅ {f} ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ {f} NOT FOUND!")
        print(f"     Expected at: {path}")
        print(f"     Please update DATA_DIR or upload the file.")

# Output directory
SAVE_DIR = "/kaggle/working/maxpooling_results"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"\nOutput directory: {SAVE_DIR}")

# %% [markdown]
# ## Step 3: Quick Data Sanity Check

# %%
import pandas as pd

# Check Zero-Day split
zd_train = pd.read_csv(os.path.join(DATA_DIR, "zero_day_train.csv"))
zd_test = pd.read_csv(os.path.join(DATA_DIR, "zero_day_test.csv"))

print("=" * 50)
print("ZERO-DAY SPLIT")
print("=" * 50)
print(f"Train samples: {len(zd_train)}")
print(f"Test samples:  {len(zd_test)}")
print(f"\nTrain family distribution:")
print(zd_train['family'].value_counts().sort_index())
print(f"\nTest family distribution:")
print(zd_test['family'].value_counts().sort_index())

# Check Standard split
std_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
std_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print("\n" + "=" * 50)
print("STANDARD (FAMILY) SPLIT")
print("=" * 50)
print(f"Train samples: {len(std_train)}")
print(f"Test samples:  {len(std_test)}")
print(f"\nTest family distribution:")
print(std_test['family'].value_counts().sort_index())

# Verify no leakage
overlap = pd.merge(zd_train, zd_test, how='inner')
print(f"\n✅ Zero-Day split leakage check: {len(overlap)} overlapping rows")
overlap2 = pd.merge(std_train, std_test, how='inner')
print(f"✅ Standard split leakage check: {len(overlap2)} overlapping rows")

# %% [markdown]
# ## Step 4: Import the Training Script

# %%
# If you uploaded srdc_maxpooling_training.py as part of your dataset,
# add it to the Python path
import sys
sys.path.insert(0, DATA_DIR)
sys.path.insert(0, "/kaggle/working")

# If the script is in your dataset, copy it to working directory
script_path = os.path.join(DATA_DIR, "srdc_maxpooling_training.py")
if os.path.exists(script_path):
    import shutil
    shutil.copy(script_path, "/kaggle/working/srdc_maxpooling_training.py")
    print("✅ Copied training script to working directory")

# %% [markdown]
# ## Step 5: Train Zero-Day Detection Model (Binary: Ransomware vs Goodware)
#
# Expected training time on T4 GPU: **~15-20 minutes** for 20 epochs.

# %%
print("=" * 70)
print("  TASK 1: ZERO-DAY RANSOMWARE DETECTION (Binary)")
print("=" * 70)

# Run the training script via Python command-line interface
!python /kaggle/working/srdc_maxpooling_training.py \
    --task zero_day \
    --data_dir {DATA_DIR} \
    --save_dir {SAVE_DIR} \
    --epochs 20

# %% [markdown]
# ## Step 6: Train Family Classification Model (12-class)
#
# Expected training time on T4 GPU: **~15-20 minutes** for 20 epochs.

# %%
print("=" * 70)
print("  TASK 2: RANSOMWARE FAMILY CLASSIFICATION (12-class)")
print("=" * 70)

# Run the training script via Python command-line interface
!python /kaggle/working/srdc_maxpooling_training.py \
    --task family \
    --data_dir {DATA_DIR} \
    --save_dir {SAVE_DIR} \
    --epochs 20

# %% [markdown]
# ## Step 7: Summary & Download

# %%
print("\n" + "=" * 70)
print("  TRAINING COMPLETE — FINAL SUMMARY")
print("=" * 70)

# List all output files
print(f"\n  All output files in {SAVE_DIR}:")
import os
for f in sorted(os.listdir(SAVE_DIR)):
    if f.endswith('.pth') or f.endswith('.csv') or f.endswith('.txt'):
        size_mb = os.path.getsize(os.path.join(SAVE_DIR, f)) / (1024 * 1024)
        print(f"    {f} ({size_mb:.1f} MB)")

print("\n" + "=" * 70)
print("  NEXT STEPS:")
print("  1. Go to your notebook output page (after commit completes).")
print("  2. Download these files:")
print("     - srdc_zero_day_BEST.pth")
print("     - srdc_family_BEST.pth")
print("  3. Place them in your local project at:")
print("     project/SRDC/result/srdc_zero_day_BEST.pth")
print("     project/SRDC/result/srdc_family_BEST.pth")
print("  4. Restart your Flask backend to load the new models!")
print("=" * 70)
