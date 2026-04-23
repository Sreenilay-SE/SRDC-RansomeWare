import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("after_feature_internal_semantic_process_data.csv")

# Ensure 'family' is string (sometimes it's numeric in your output)
df['family'] = df['family'].astype(str)

# Binary label for stratification
df['is_ransomware'] = (df['family'] != 'Goodware').astype(int)

# 80/20 stratified split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['is_ransomware'],
    random_state=42  # or 123 for different
)

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("New split created:")
print("Train:", train_df.shape, train_df['family'].value_counts())
print("Test:", test_df.shape, test_df['family'].value_counts())