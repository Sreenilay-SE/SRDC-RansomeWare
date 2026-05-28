"""
AUDIT 2 -- Model Behavior: Fresh Inference on srdc_zero_day_BEST.pth
======================================================================
Checks:
  2A  Confusion matrix + per-class precision/recall/F1
  2B  Confidence score distribution (buckets)
  2C  Majority-class baseline comparison
  2D  Random label test (100 shuffles)
  2E  Zero-day family spot check (qualitative)
  3C  Token truncation rate (checks how many inputs > 1024 tokens)

Run from project/SRDC/:
  python audit2_model.py
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import Counter

BASE     = os.path.dirname(os.path.abspath(__file__))
ZD_TRAIN = os.path.join(BASE, 'splits', 'zero_day_train.csv')
ZD_TEST  = os.path.join(BASE, 'splits', 'zero_day_test.csv')
MODEL_ZD = os.path.join(BASE, 'result', 'srdc_zero_day_BEST.pth')

ZERO_DAY_FAMILIES = {
    '8':  'PGPCODER',
    '9':  'Reveton',
    '10': 'TeslaCrypt',
    '11': 'Trojan-Ransom',
}

# ─────────────────────────────────────────────────────────────────────────────
print('=' * 70)
print('AUDIT 2 -- Model Behavior (Fresh Inference)')
print('=' * 70)

# ── Import check ─────────────────────────────────────────────────────────────
try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset as TorchDataset, DataLoader
    from transformers import GPT2Tokenizer, GPT2Model
    from sklearn.metrics import (classification_report, accuracy_score,
                                 confusion_matrix, recall_score)
    print(f'torch      : {torch.__version__}')
    print(f'CUDA       : {torch.cuda.is_available()}')
except ImportError as e:
    print(f'MISSING DEPENDENCY: {e}')
    print('Install: pip install torch transformers scikit-learn')
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  Dataset (mirrors training code exactly)
# ─────────────────────────────────────────────────────────────────────────────
class ZeroDayDataset(TorchDataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.texts = (
            self.df['apiFeatures'].fillna('') + ' ' +
            self.df['dropFeatures'].fillna('') + ' ' +
            self.df['regFeatures'].fillna('') + ' ' +
            self.df['filesFeatures'].fillna('') + ' ' +
            self.df['filesEXTFeatures'].fillna('') + ' ' +
            self.df['dirFeatures'].fillna('') + ' ' +
            self.df['strFeatures'].fillna('')
        ).str.strip().tolist()
        # Label: 0=Goodware (family '0'), 1=Ransomware (any other family)
        self.labels = (self.df['family'].astype(str) != '0').astype(int).tolist()
        self.families = self.df['family'].astype(str).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc  = self.tokenizer(
            text, truncation=True, max_length=1024,
            padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long),
            'family':         self.families[idx],
            'text':           text,
        }

# ─────────────────────────────────────────────────────────────────────────────
#  Classifier (must match saved architecture)
# ─────────────────────────────────────────────────────────────────────────────
class Classifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super().__init__()
        self.gpt    = GPT2Model.from_pretrained('zhouce/RDC-GPT')
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out    = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state.mean(dim=1)
        return self.linear(pooled)

# ─────────────────────────────────────────────────────────────────────────────
#  Load model weights
# ─────────────────────────────────────────────────────────────────────────────
print(f'\nLoading model: {MODEL_ZD}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model = Classifier()
try:
    state = torch.load(MODEL_ZD, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print('Model loaded successfully.')
except Exception as e:
    print(f'ERROR loading model: {e}')
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  Load data
# ─────────────────────────────────────────────────────────────────────────────
zd_train = pd.read_csv(ZD_TRAIN)
zd_test  = pd.read_csv(ZD_TEST)
zd_train['family'] = zd_train['family'].astype(str)
zd_test['family']  = zd_test['family'].astype(str)

print(f'\nTest set size: {len(zd_test)} samples')
test_ds     = ZeroDayDataset(zd_test)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                         collate_fn=lambda batch: {
                             'input_ids':      torch.stack([b['input_ids'] for b in batch]),
                             'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
                             'label':          torch.stack([b['label'] for b in batch]),
                             'family':         [b['family'] for b in batch],
                             'text':           [b['text'] for b in batch],
                         })

# ─────────────────────────────────────────────────────────────────────────────
#  Fresh inference
# ─────────────────────────────────────────────────────────────────────────────
print('\nRunning fresh inference on zero_day_test.csv ...')
all_preds    = []
all_trues    = []
all_confs    = []
all_families = []
all_texts    = []

with torch.no_grad():
    for b_idx, batch in enumerate(test_loader):
        inp  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labs = batch['label'].to(device)

        logits = model(inp, mask)
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)
        confs  = probs.max(dim=1).values

        all_preds.extend(preds.cpu().tolist())
        all_trues.extend(labs.cpu().tolist())
        all_confs.extend(confs.cpu().tolist())
        all_families.extend(batch['family'])
        all_texts.extend(batch['text'])

        done = min((b_idx + 1) * 4, len(zd_test))
        print(f'  [{done}/{len(zd_test)}]', end='\r', flush=True)

print()

all_preds = np.array(all_preds)
all_trues = np.array(all_trues)
all_confs = np.array(all_confs)

srdc_acc = accuracy_score(all_trues, all_preds)
print(f'\nSRDC Zero-Day Accuracy (fresh inference): {srdc_acc:.4f}  ({srdc_acc*100:.2f}%)')

# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 2A -- Confusion matrix + per-class report
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 70)
print('CHECK 2A -- Confusion Matrix & Per-class Report')
print('=' * 70)

cm = confusion_matrix(all_trues, all_preds)
print('\nConfusion Matrix:')
print(f'{"":>22} Pred: Goodware   Pred: Ransomware')
print(f'{"True: Goodware":>22}    {cm[0,0]:>7}          {cm[0,1]:>7}')
print(f'{"True: Ransomware":>22}    {cm[1,0]:>7}          {cm[1,1]:>7}')

print('\nClassification Report:')
report = classification_report(all_trues, all_preds,
                                target_names=['Goodware', 'Ransomware'], digits=4)
print(report)

recalls = recall_score(all_trues, all_preds, average=None, zero_division=0)
print(f'Per-class recalls: Goodware={recalls[0]:.4f}, Ransomware={recalls[1]:.4f}')

if any(r > 0.98 for r in recalls) and any(r < 0.50 for r in recalls):
    print('WARNING: DEGENERATE MODEL -- one class recall > 0.98 while another < 0.50')
    print('         This suggests the model may be collapsing to one class.')
else:
    print('No degenerate prediction pattern detected.')

pred_counts = Counter(all_preds.tolist())
print(f'\nPrediction distribution: Goodware={pred_counts[0]}, Ransomware={pred_counts[1]}')
maj_pct = max(pred_counts.values()) / len(all_preds)
if maj_pct > 0.95:
    print(f'WARNING: {maj_pct*100:.1f}% of all predictions are the same class!')

# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 2B -- Confidence score distribution
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 70)
print('CHECK 2B -- Confidence Score Distribution')
print('=' * 70)

buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.001)]
labels  = ['[0.5-0.6]', '[0.6-0.7]', '[0.7-0.8]', '[0.8-0.9]', '[0.9-1.0]']
n = len(all_confs)

print(f'\n{"Bucket":>12}  {"Count":>7}  {"Percentage":>10}')
print('-' * 35)
high_conf_pct = 0.0
for (lo, hi), lbl in zip(buckets, labels):
    cnt = int(np.sum((all_confs >= lo) & (all_confs < hi)))
    pct = cnt / n * 100
    print(f'{lbl:>12}  {cnt:>7}  {pct:>9.1f}%')
    if lbl == '[0.9-1.0]':
        high_conf_pct = pct

print(f'\nMean confidence : {all_confs.mean():.4f}')
print(f'Median confidence: {np.median(all_confs):.4f}')
print(f'Min: {all_confs.min():.4f}  Max: {all_confs.max():.4f}')

if high_conf_pct > 60:
    print(f'\nPASS: {high_conf_pct:.1f}% of predictions in high-confidence [0.9-1.0] bucket.')
else:
    print(f'\nWARNING: Only {high_conf_pct:.1f}% of predictions in [0.9-1.0]. Model may be uncertain.')

# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 2C -- Majority-class baseline
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 70)
print('CHECK 2C -- Majority-Class Baseline Comparison')
print('=' * 70)

train_labels  = (zd_train['family'].astype(str) != '0').astype(int)
majority_cls  = int(train_labels.mode()[0])
baseline_pred = np.full(len(all_trues), majority_cls)
baseline_acc  = accuracy_score(all_trues, baseline_pred)
improvement   = (srdc_acc - baseline_acc) * 100

print(f'\nMajority class in training set: {majority_cls} ({"Ransomware" if majority_cls == 1 else "Goodware"})')
print(f'Baseline accuracy : {baseline_acc*100:.2f}%')
print(f'SRDC accuracy     : {srdc_acc*100:.2f}%')
print(f'Improvement       : {improvement:.2f} percentage points')

if improvement > 10:
    print(f'PASS: SRDC outperforms baseline by {improvement:.2f} pp (> 10 pp threshold).')
else:
    print(f'FAIL: SRDC only beats baseline by {improvement:.2f} pp (threshold: 10 pp).')

# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 2D -- Random label test (100 shuffles)
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 70)
print('CHECK 2D -- Random Label Test (100 shuffles)')
print('=' * 70)

rng = np.random.default_rng(0)
shuffled_accs = [accuracy_score(rng.permutation(all_trues), all_preds)
                 for _ in range(100)]

mean_sh = np.mean(shuffled_accs)
std_sh  = np.std(shuffled_accs)
z_score = (srdc_acc - mean_sh) / std_sh if std_sh > 0 else float('inf')

print(f'\nShuffled accuracy (100 runs):')
print(f'  Mean : {mean_sh:.4f}')
print(f'  Std  : {std_sh:.6f}')
print(f'SRDC accuracy : {srdc_acc:.4f}')
print(f'Z-score       : {z_score:.2f}')

if z_score > 3.0:
    print(f'PASS: SRDC is {z_score:.2f} standard deviations above chance (> 3.0 threshold).')
else:
    print(f'FAIL: SRDC is only {z_score:.2f}sigma above chance (threshold: 3.0).')

# ─────────────────────────────────────────────────────────────────────────────
#  CHECK 2E -- Zero-day family spot check
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 70)
print('CHECK 2E -- Zero-Day Family Spot Check (up to 3 samples per family)')
print('=' * 70)

families_idx = {}
for i, fam in enumerate(all_families):
    if fam in ZERO_DAY_FAMILIES:
        families_idx.setdefault(fam, []).append(i)

print(f'\n{"Family":>14}  {"Sample#":>8}  {"True Label":>12}  {"Prediction":>12}  {"Confidence":>10}  {"Correct":>7}')
print('-' * 75)

per_family_acc = {}
for fam in sorted(ZERO_DAY_FAMILIES.keys()):
    fname   = ZERO_DAY_FAMILIES[fam]
    indices = families_idx.get(fam, [])
    if not indices:
        print(f'{fname:>14}  NO SAMPLES IN TEST SET')
        continue
    for i in indices[:3]:
        t = all_trues[i]
        p = all_preds[i]
        c = all_confs[i]
        t_str = 'Ransomware' if t == 1 else 'Goodware'
        p_str = 'Ransomware' if p == 1 else 'Goodware'
        ok    = 'YES' if t == p else 'NO'
        print(f'{fname:>14}  {i:>8}  {t_str:>12}  {p_str:>12}  {c:>10.4f}  {ok:>7}')

    fam_correct = sum(all_trues[i] == all_preds[i] for i in indices)
    fam_acc = fam_correct / len(indices)
    per_family_acc[fname] = (fam_acc, len(indices))

print()
print('Per-family accuracy on zero-day test samples:')
worst_fam, worst_acc = None, 1.0
for fname, (acc, n_s) in sorted(per_family_acc.items()):
    print(f'  {fname:>14}: {acc*100:.1f}%  (n={n_s})')
    if acc < worst_acc:
        worst_acc, worst_fam = acc, fname
if worst_fam:
    print(f'\nWorst-performing zero-day family: {worst_fam} ({worst_acc*100:.1f}%)')

# ─────────────────────────────────────────────────────────────────────────────
#  BUG 3C (inline) -- Token truncation rate
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 70)
print('BUG 3C -- Token Truncation Rate (> 1024 tokens)')
print('=' * 70)

from transformers import GPT2Tokenizer as _GPT2Tok
tok = _GPT2Tok.from_pretrained('gpt2')
tok.pad_token = tok.eos_token

sample_texts  = all_texts[:500]
truncated_cnt = sum(
    1 for t in sample_texts
    if len(tok.encode(str(t), add_special_tokens=False)) > 1024
)
sample_n   = len(sample_texts)
trunc_pct  = truncated_cnt / sample_n * 100
print(f'\nChecked {sample_n} test samples.')
print(f'Samples exceeding 1024 tokens: {truncated_cnt} ({trunc_pct:.1f}%)')

# Also print token length statistics
token_lens = [len(tok.encode(str(t), add_special_tokens=False)) for t in sample_texts]
print(f'Token length stats:')
print(f'  Mean   : {np.mean(token_lens):.0f}')
print(f'  Median : {np.median(token_lens):.0f}')
print(f'  Max    : {max(token_lens)}')
print(f'  Min    : {min(token_lens)}')

if trunc_pct < 20:
    print(f'PASS: Truncation rate {trunc_pct:.1f}% is below 20% threshold.')
else:
    print(f'FAIL: Truncation rate {trunc_pct:.1f}% EXCEEDS 20% -- model receives incomplete inputs!')

print()
print('=' * 70)
print('AUDIT 2 COMPLETE')
print('=' * 70)
