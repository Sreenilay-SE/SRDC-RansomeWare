"""
FRESH AUDIT 1 — Data Integrity: Leakage Check
Written from scratch — no reference to existing audit files.
Author: ML Auditor (Antigravity)
"""

import pandas as pd
import hashlib
import sys

BASE = r"C:\Users\sree nilay\Downloads\DOMAIN-PRO-SRDC\DOMAIN-PRO-SRDC\project\SRDC"

ZD_TRAIN = f"{BASE}\\splits\\zero_day_train.csv"
ZD_TEST  = f"{BASE}\\splits\\zero_day_test.csv"
FULL_DS  = f"{BASE}\\after_feature_internal_semantic_process_data.csv"

print("=" * 70)
print("AUDIT 1 — DATA INTEGRITY & LEAKAGE CHECK")
print("=" * 70)

# -------------------------------------------------------
# 1A — Family-level leakage check
# -------------------------------------------------------
print("\n" + "─" * 70)
print("AUDIT 1A — Family-Level Leakage Check")
print("─" * 70)

train_zd = pd.read_csv(ZD_TRAIN)
test_zd  = pd.read_csv(ZD_TEST)

print(f"\n[INFO] zero_day_train.csv columns:\n  {list(train_zd.columns)}")
print(f"\n[INFO] zero_day_test.csv  columns:\n  {list(test_zd.columns)}")

# Detect label column: prefer 'family', then 'is_ransomware'
if 'family' in train_zd.columns:
    label_col = 'family'
elif 'is_ransomware' in train_zd.columns:
    label_col = 'is_ransomware'
else:
    print("[ERROR] No recognizable label column found!")
    sys.exit(1)

print(f"\n[INFO] Using label column: '{label_col}'")

train_families = set(train_zd[label_col].astype(str).unique())
test_families  = set(test_zd[label_col].astype(str).unique())

print(f"\n[INFO] Unique values in TRAIN: {sorted(train_families)}")
print(f"[INFO] Unique values in TEST:  {sorted(test_families)}")

intersection = train_families & test_families
if len(intersection) == 0:
    print("\n[AUDIT 1A] ✅  PASS — No family-level leakage detected.")
    print("           The label sets are disjoint between train and test.")
else:
    print(f"\n[AUDIT 1A] ❌  FAIL — FAMILY LEAKAGE DETECTED!")
    print(f"           Overlapping label values: {sorted(intersection)}")

# -------------------------------------------------------
# 1B — Sample-level leakage check
# -------------------------------------------------------
print("\n" + "─" * 70)
print("AUDIT 1B — Sample-Level Leakage Check (MD5 fingerprint)")
print("─" * 70)

FEATURE_COLS = ['apiFeatures', 'dropFeatures', 'regFeatures',
                'filesFeatures', 'filesEXTFeatures', 'dirFeatures', 'strFeatures']

available_cols_train = [c for c in FEATURE_COLS if c in train_zd.columns]
available_cols_test  = [c for c in FEATURE_COLS if c in test_zd.columns]

print(f"[INFO] Feature columns found in train: {available_cols_train}")
print(f"[INFO] Feature columns found in test:  {available_cols_test}")

shared_cols = [c for c in available_cols_train if c in available_cols_test]

def make_fingerprint(row, cols):
    combined = "|".join(str(row[c]) for c in cols)
    return hashlib.md5(combined.encode('utf-8', errors='replace')).hexdigest()

print(f"\n[INFO] Computing MD5 fingerprints for {len(train_zd)} train rows...")
train_zd['_fp'] = train_zd.apply(lambda r: make_fingerprint(r, shared_cols), axis=1)
print(f"[INFO] Computing MD5 fingerprints for {len(test_zd)} test rows...")
test_zd['_fp']  = test_zd.apply(lambda r: make_fingerprint(r, shared_cols), axis=1)

train_hashes = set(train_zd['_fp'])
test_hashes  = set(test_zd['_fp'])
overlap_hashes = train_hashes & test_hashes

if len(overlap_hashes) == 0:
    print("\n[AUDIT 1B] ✅  PASS — No sample-level duplication detected.")
    print(f"           All {len(train_zd)} train hashes are unique from {len(test_zd)} test hashes.")
else:
    dup_count = sum(train_zd['_fp'].isin(overlap_hashes)) + sum(test_zd['_fp'].isin(overlap_hashes))
    print(f"\n[AUDIT 1B] ❌  FAIL — SAMPLE LEAKAGE DETECTED!")
    print(f"           {len(overlap_hashes)} unique fingerprints appear in both train and test.")
    print(f"           Approximately {dup_count} total rows are duplicated across splits.")

# -------------------------------------------------------
# 1C — Family distribution table
# -------------------------------------------------------
print("\n" + "─" * 70)
print("AUDIT 1C — Family Distribution Table")
print("─" * 70)

all_labels = sorted(
    set(train_zd[label_col].astype(str).unique()) |
    set(test_zd[label_col].astype(str).unique())
)

train_counts = train_zd[label_col].astype(str).value_counts().to_dict()
test_counts  = test_zd[label_col].astype(str).value_counts().to_dict()

print(f"\n{'Label Value':<20} {'Train Count':>12} {'Test Count':>12}")
print("-" * 46)
for lbl in all_labels:
    tc = train_counts.get(lbl, 0)
    sc = test_counts.get(lbl, 0)
    print(f"{lbl:<20} {tc:>12} {sc:>12}")

# Specifically call out zero-day families
print("\n[INFO] Expected zero-day families (should have 0 in train):")
ZERO_DAY_FAMILIES = ['PGPCODER', 'Reveton', 'TeslaCrypt', 'Trojan-Ransom',
                     '8', '9', '10', '11']  # also check numeric equivalents
for fam in ZERO_DAY_FAMILIES:
    tc = train_counts.get(str(fam), 0)
    sc = test_counts.get(str(fam), 0)
    if tc > 0:
        print(f"  ❌ FAIL: '{fam}' has {tc} samples in TRAIN — zero-day contamination!")
    elif sc > 0:
        print(f"  ✅  PASS: '{fam}' → train: 0 | test: {sc}")

# -------------------------------------------------------
# 1D — Full dataset consistency check
# -------------------------------------------------------
print("\n" + "─" * 70)
print("AUDIT 1D — Full Dataset Consistency Check")
print("─" * 70)

print(f"\n[INFO] Loading full dataset: {FULL_DS}")
full_df = pd.read_csv(FULL_DS)

print(f"\n[INFO] Full dataset columns:\n  {list(full_df.columns)}")
print(f"\n[INFO] Full dataset shape: {full_df.shape}")

if label_col in full_df.columns:
    full_label_counts = full_df[label_col].astype(str).value_counts().sort_index()
    print(f"\n[INFO] Label value counts in full dataset:")
    for lbl, cnt in full_label_counts.items():
        print(f"  {lbl:<20} {cnt:>8}")
    total_full = len(full_df)
else:
    # try 'family'
    if 'family' in full_df.columns:
        full_label_counts = full_df['family'].astype(str).value_counts().sort_index()
        print(f"\n[INFO] 'family' column counts in full dataset:")
        for lbl, cnt in full_label_counts.items():
            print(f"  {lbl:<20} {cnt:>8}")
    total_full = len(full_df)

total_train = len(train_zd)
total_test  = len(test_zd)
total_splits = total_train + total_test

print(f"\n[INFO] Full dataset rows:  {total_full}")
print(f"[INFO] Train rows:         {total_train}")
print(f"[INFO] Test rows:          {total_test}")
print(f"[INFO] Train + Test total: {total_splits}")

if total_splits == total_full:
    print(f"\n[AUDIT 1D] ✅  PASS — Train + Test ({total_splits}) == Full dataset ({total_full}).")
elif abs(total_splits - total_full) <= 10:
    print(f"\n[AUDIT 1D] ⚠️  WARN — Minor discrepancy: {total_splits} vs {total_full} (diff={abs(total_splits-total_full)}).")
    print("           This may be due to duplicate removal during split creation.")
else:
    print(f"\n[AUDIT 1D] ❌  FAIL — Significant mismatch: {total_splits} vs {total_full}.")
    print(f"           {abs(total_splits - total_full)} rows unaccounted for.")

print("\n" + "=" * 70)
print("AUDIT 1 COMPLETE")
print("=" * 70)
