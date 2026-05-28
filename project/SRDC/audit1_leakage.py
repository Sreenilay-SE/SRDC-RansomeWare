import pandas as pd
import hashlib
import os

BASE = os.path.dirname(os.path.abspath(__file__))
ZD_TRAIN = os.path.join(BASE, 'splits', 'zero_day_train.csv')
ZD_TEST  = os.path.join(BASE, 'splits', 'zero_day_test.csv')

ZERO_DAY_FAMILIES = {
    '8':  'PGPCODER',
    '9':  'Reveton',
    '10': 'TeslaCrypt',
    '11': 'Trojan-Ransom',
}

def row_hash(row):
    return hashlib.sha256('|'.join(str(v) for v in row.values).encode()).hexdigest()

print('=' * 65)
print('AUDIT 1 -- DATA INTEGRITY')
print('=' * 65)

zd_train = pd.read_csv(ZD_TRAIN)
zd_test  = pd.read_csv(ZD_TEST)
zd_train['family'] = zd_train['family'].astype(str)
zd_test['family']  = zd_test['family'].astype(str)

print(f'zero_day_train rows: {len(zd_train)}')
print(f'zero_day_test  rows: {len(zd_test)}')
print(f'zero_day_train columns: {list(zd_train.columns)}')

# 1A  Family-level leakage
train_fams = set(zd_train['family'].unique())
test_fams  = set(zd_test['family'].unique())
overlap    = train_fams & test_fams

print()
print('Train unique families:', sorted(train_fams))
print('Test  unique families:', sorted(test_fams))
print('Intersection (should be empty for valid zero-day):', sorted(overlap))

print()
if not overlap:
    print('LEAKAGE CHECK PASSED -- no overlap between train and test families')
else:
    names = [ZERO_DAY_FAMILIES.get(f, f) for f in overlap]
    print('LEAKAGE DETECTED -- families in BOTH train and test:', sorted(overlap))
    print('Human names:', names)

# Confirm zero-day families absent from train
zd_in_train = [f for f in ZERO_DAY_FAMILIES if f in train_fams]
print()
if zd_in_train:
    bad = [ZERO_DAY_FAMILIES[f] for f in zd_in_train]
    print('CRITICAL: Zero-day families FOUND in training set!', bad)
else:
    print('CONFIRMED: All 4 zero-day families (8=PGPCODER, 9=Reveton, 10=TeslaCrypt, 11=Trojan-Ransom)')
    print('           are ABSENT from the training set.')

# 1B  Sample-level hash check
print()
print('Computing SHA-256 hashes for sample-level leakage...')
train_hashes = set(row_hash(r) for _, r in zd_train.iterrows())
test_dups = sum(1 for _, r in zd_test.iterrows() if row_hash(r) in train_hashes)
print()
if test_dups == 0:
    print('SAMPLE-LEVEL LEAKAGE CHECK PASSED -- 0 duplicate samples between train and test')
else:
    print(f'SAMPLE-LEVEL LEAKAGE DETECTED -- {test_dups} test samples found verbatim in training!')

# 1C  Family distribution table
print()
print('Family Distribution Table:')
print(f'{"Family":>18} | {"Train Count":>11} | {"Test Count":>10} | Notes')
print('-' * 65)
tc = zd_train['family'].value_counts().to_dict()
ec = zd_test['family'].value_counts().to_dict()
for fam in sorted(train_fams | test_fams):
    name = ZERO_DAY_FAMILIES.get(fam, f'Family_{fam}')
    note = ' <- ZERO-DAY (held-out)' if fam in ZERO_DAY_FAMILIES else ''
    print(f'{name:>18} | {tc.get(fam, 0):>11} | {ec.get(fam, 0):>10}{note}')

print()
print('Done.')
