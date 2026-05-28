import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Set paths
DATA_PATH = "after_feature_internal_semantic_process_data.csv"
SPLIT_DIR = "clean_splits"
os.makedirs(SPLIT_DIR, exist_ok=True)

print("Loading original dataset...")
df = pd.read_csv(DATA_PATH)
df['family'] = df['family'].astype(str)

# 1. Clean Data: Remove exact duplicate rows to prevent leakage
initial_len = len(df)
df = df.drop_duplicates()
print(f"Dropped {initial_len - len(df)} duplicate rows. Clean dataset size: {len(df)}")

# Create binary label for stratification
df['is_ransomware'] = (df['family'] != '0').astype(int)

# ==========================================
# 2. Fix Standard Split (Train/Test)
# ==========================================
print("\n--- Generating Standard Split ---")
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['is_ransomware'],
    random_state=42
)

# Verify no overlap
overlap_standard = pd.merge(train_df, test_df, how='inner')
print(f"Standard Split Leakage (overlapping rows): {len(overlap_standard)}")

train_df.to_csv(os.path.join(SPLIT_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(SPLIT_DIR, "test.csv"), index=False)
print("Saved train.csv and test.csv")

# ==========================================
# 3. Fix Zero-Day Split
# ==========================================
print("\n--- Generating Zero-Day Split ---")
# Separate Goodware and Ransomware
goodware_df = df[df['family'] == '0']
ransomware_df = df[df['family'] != '0']

# Split Goodware into train and test without overlap (80/20)
gw_train, gw_test = train_test_split(goodware_df, test_size=0.2, random_state=42)

# Define Zero-Day Families based on current setup
train_families = ['1', '2', '3', '4', '5', '6', '7']
test_families = ['8', '9', '10', '11']

# Split Ransomware based on families
rw_train = ransomware_df[ransomware_df['family'].isin(train_families)]
rw_test = ransomware_df[ransomware_df['family'].isin(test_families)]

# Combine Goodware and Ransomware for final sets
zero_day_train = pd.concat([gw_train, rw_train]).sample(frac=1, random_state=42) # Shuffle
zero_day_test = pd.concat([gw_test, rw_test]).sample(frac=1, random_state=42) # Shuffle

# Verify no overlap
overlap_zero_day = pd.merge(zero_day_train, zero_day_test, how='inner')
print(f"Zero-Day Split Leakage (overlapping rows): {len(overlap_zero_day)}")

zero_day_train.to_csv(os.path.join(SPLIT_DIR, "zero_day_train.csv"), index=False)
zero_day_test.to_csv(os.path.join(SPLIT_DIR, "zero_day_test.csv"), index=False)
print("Saved zero_day_train.csv and zero_day_test.csv")
print("\nAll splits regenerated successfully without data leakage!")
