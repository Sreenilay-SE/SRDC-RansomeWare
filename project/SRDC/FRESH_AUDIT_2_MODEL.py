"""
FRESH AUDIT 2 — Model Behavior: Optimized for CPU
Uses torch.inference_mode, float32 with all CPU threads, batch_size=8.
Writes predictions incrementally to audit2_predictions.csv as it goes.
Author: ML Auditor (Antigravity)
"""

import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import torch
from torch import nn

torch.set_num_threads(10)  # Use all available CPU threads

BASE    = r"C:\Users\sree nilay\Downloads\DOMAIN-PRO-SRDC\DOMAIN-PRO-SRDC\project\SRDC"
ZD_TEST = f"{BASE}\\splits\\zero_day_test.csv"
ZD_TRAIN= f"{BASE}\\splits\\zero_day_train.csv"
FAM_TEST= f"{BASE}\\splits\\test.csv"
ZD_MODEL  = f"{BASE}\\result\\srdc_zero_day_BEST.pth"
FAM_MODEL = f"{BASE}\\result\\srdc_family_BEST.pth"
PRED_OUT  = f"{BASE}\\audit2_zd_predictions.csv"
FAM_OUT   = f"{BASE}\\audit2_fam_predictions.csv"

FAMILY_NAMES = {
    0: 'Goodware', 1: 'Citroni', 2: 'CryptLocker',
    3: 'CryptoWall', 4: 'Kollah', 5: 'Kovter',
    6: 'Locker', 7: 'Matsnu', 8: 'PGPCODER',
    9: 'Reveton', 10: 'TeslaCrypt', 11: 'Trojan-Ransom'
}
ZD_CLASS_NAMES = ['Goodware', 'Ransomware']

print("=" * 70)
print("AUDIT 2 — MODEL BEHAVIOR VERIFICATION (CPU-Optimised)")
print("=" * 70)

from transformers import GPT2Tokenizer, GPT2Model
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, recall_score)

print(f"\n[INFO] Loading GPT-2 tokenizer (cached)...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

class Classifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained("zhouce/RDC-GPT")
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.linear(pooled)

def get_text(row):
    return (
        str(row.get('apiFeatures', '')) + " " +
        str(row.get('dropFeatures', '')) + " " +
        str(row.get('regFeatures', '')) + " " +
        str(row.get('filesFeatures', '')) + " " +
        str(row.get('filesEXTFeatures', '')) + " " +
        str(row.get('dirFeatures', '')) + " " +
        str(row.get('strFeatures', ''))
    ).strip()

def run_inference(model, df, out_csv, batch_size=8, label_col='is_ransomware'):
    """Run inference with incremental CSV writing. Returns (preds, confs, probs_list)."""
    texts = [get_text(row) for _, row in df.iterrows()]
    true_labels = df[label_col].astype(int).tolist()
    
    all_preds, all_confs, all_probs = [], [], []
    rows_out = []
    
    model.eval()
    total = len(texts)
    t0 = time.time()
    
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            batch_texts = texts[start:start + batch_size]
            enc = tokenizer(
                batch_texts, truncation=True, max_length=1024,
                padding='max_length', return_tensors='pt'
            )
            logits = model(enc['input_ids'], enc['attention_mask'])
            probs_t = torch.softmax(logits, dim=1).numpy()
            preds_t = np.argmax(probs_t, axis=1)
            confs_t = probs_t[np.arange(len(preds_t)), preds_t]
            
            all_preds.extend(preds_t.tolist())
            all_confs.extend(confs_t.tolist())
            all_probs.extend(probs_t.tolist())
            
            for i, idx in enumerate(range(start, min(start + batch_size, total))):
                rows_out.append({
                    'idx': idx,
                    'true_label': true_labels[idx],
                    'pred': int(preds_t[i]),
                    'confidence': float(confs_t[i]),
                    'family': df.iloc[idx].get('family', -1)
                })
            
            done = min(start + batch_size, total)
            elapsed = time.time() - t0
            eta = (elapsed / done) * (total - done) if done > 0 else 0
            print(f"  [{done}/{total}] elapsed={elapsed:.1f}s  ETA={eta:.1f}s", flush=True)
            
            # Write incrementally
            pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    
    return all_preds, all_confs, all_probs, true_labels

# -------------------------------------------------------
# Zero-Day Binary Model
# -------------------------------------------------------
print(f"\n[INFO] Loading test data ({ZD_TEST})...")
test_zd = pd.read_csv(ZD_TEST).fillna('')
print(f"[INFO] {len(test_zd)} samples")

print(f"\n[INFO] Loading zero-day model (498MB)...")
t_load = time.time()
zd_model = Classifier(hidden_size=768, num_classes=2)
sd = torch.load(ZD_MODEL, map_location='cpu')
zd_model.load_state_dict(sd)
zd_model.eval()
print(f"[INFO] Model loaded in {time.time()-t_load:.1f}s")

print(f"\n[INFO] Running inference on {len(test_zd)} samples (batch_size=8)...")
preds_zd, confs_zd, probs_zd, true_zd = run_inference(
    zd_model, test_zd, PRED_OUT, batch_size=8, label_col='is_ransomware'
)
print(f"\n[INFO] Inference complete. Predictions saved to {PRED_OUT}")

# Free model memory immediately
del zd_model, sd
import gc; gc.collect()

# ─── AUDIT 2A ────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("AUDIT 2A — Confusion Matrix & Classification Report")
print("-" * 70)

acc = accuracy_score(true_zd, preds_zd)
cm  = confusion_matrix(true_zd, preds_zd)
rpt = classification_report(true_zd, preds_zd,
                             target_names=ZD_CLASS_NAMES, digits=4)
print(f"\nOverall Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
print(f"\nConfusion Matrix (rows=True, cols=Predicted):")
print(f"              Goodware  Ransomware")
print(f"  Goodware       {cm[0,0]:5}       {cm[0,1]:5}")
print(f"  Ransomware     {cm[1,0]:5}       {cm[1,1]:5}")
print(f"\n{rpt}")

recalls = recall_score(true_zd, preds_zd, average=None)
print(f"[INFO] Per-class recall — Goodware: {recalls[0]:.4f} | Ransomware: {recalls[1]:.4f}")
if max(recalls) >= 0.98 and min(recalls) < 0.50:
    print(f"[ALERT] CLASS IMBALANCE IN RECALL: max={max(recalls):.2%} vs min={min(recalls):.2%}")
else:
    print(f"[OK] No severe recall imbalance.")

# ─── AUDIT 2B ────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("AUDIT 2B — Confidence Score Distribution")
print("-" * 70)

buckets = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
bucket_labels = ['[0.50-0.60]', '[0.60-0.70]', '[0.70-0.80]', '[0.80-0.90]', '[0.90-1.00]']
N = len(confs_zd)

print(f"\n{'Bucket':<14} {'Count':>8} {'Pct':>8}")
print("-" * 32)
high_conf_pct = 0.0
for (lo, hi), lbl in zip(buckets, bucket_labels):
    cnt = sum(1 for c in confs_zd if lo <= c < hi)
    pct = 100.0 * cnt / N
    if lbl == '[0.90-1.00]':
        high_conf_pct = pct
    print(f"{lbl:<14} {cnt:>8} {pct:>7.1f}%")

mean_c = np.mean(confs_zd)
print(f"\n  Mean confidence:   {mean_c:.4f}")
print(f"  Median:            {np.median(confs_zd):.4f}")
print(f"  Std dev:           {np.std(confs_zd):.4f}")
n_low = sum(1 for c in confs_zd if c < 0.65)
print(f"  Below 0.65 (guessing zone): {n_low} ({100.*n_low/N:.1f}%)")
print(f"  Above 0.90 (high certainty): {sum(1 for c in confs_zd if c>0.90)} ({high_conf_pct:.1f}%)")

if high_conf_pct > 70:
    print("[INTERPRETATION] Clusters in [0.90-1.00] — genuine learned features.")
elif n_low / N > 0.40:
    print("[INTERPRETATION] WARNING: Many in guessing range [0.50-0.65]. May not have converged.")
else:
    print("[INTERPRETATION] Mixed confidence distribution.")

# ─── AUDIT 2C ────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("AUDIT 2C — Majority-Class Baseline")
print("-" * 70)

train_zd = pd.read_csv(ZD_TRAIN)
maj_cls  = int(train_zd['is_ransomware'].mode()[0])
maj_name = ZD_CLASS_NAMES[maj_cls]
maj_cnt  = (train_zd['is_ransomware'] == maj_cls).sum()
baseline_acc = accuracy_score(true_zd, [maj_cls] * len(true_zd))
diff = acc - baseline_acc

print(f"\n  Majority class in train: '{maj_name}' ({maj_cnt}/{len(train_zd)} = {100.*maj_cnt/len(train_zd):.1f}%)")
print(f"  Baseline accuracy (always predict '{maj_name}'): {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
print(f"  SRDC model accuracy:                             {acc:.4f} ({acc*100:.2f}%)")
print(f"  Difference:                                      {diff:+.4f} ({diff*100:+.2f} pts)")

if abs(diff) < 0.10:
    print(f"\n[AUDIT 2C] CRITICAL — only {diff*100:+.1f}% above trivial baseline!")
else:
    print(f"\n[AUDIT 2C] PASS — {diff*100:+.1f}% above trivial baseline.")

# ─── AUDIT 2D ────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("AUDIT 2D — Random Chance Test (200 shuffles)")
print("-" * 70)

rng = np.random.default_rng(seed=42)
true_arr = np.array(true_zd)
shuffled_accs = [accuracy_score(rng.permutation(true_arr), preds_zd) for _ in range(200)]
sh_mean = np.mean(shuffled_accs)
sh_std  = np.std(shuffled_accs)
z       = (acc - sh_mean) / (sh_std + 1e-12)

print(f"\n  Shuffled label accuracy — Mean: {sh_mean:.4f} | Std: {sh_std:.4f}")
print(f"  SRDC model accuracy:           {acc:.4f}")
print(f"  Z-score vs random:             {z:.2f} std devs above shuffled mean")

if z > 3.0:
    print(f"\n[AUDIT 2D] PASS — {z:.1f}σ above random. Result is statistically significant.")
else:
    print(f"\n[AUDIT 2D] CRITICAL — only {z:.1f}σ above random. May be within chance range.")

# ─── AUDIT 2E ────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("AUDIT 2E — Zero-Day Family Spot Check")
print("-" * 70)

test_zd2 = pd.read_csv(ZD_TEST).fillna('')
test_zd2['_pred'] = preds_zd
test_zd2['_conf'] = confs_zd
test_zd2['_ok']   = (test_zd2['_pred'] == test_zd2['is_ransomware'].astype(int))

ZD_FAMS = {'8': 'PGPCODER', '9': 'Reveton', '10': 'TeslaCrypt', '11': 'Trojan-Ransom'}
fam_accs = {}

print(f"\n{'Family':<16} {'#':>4} {'Pred':>12} {'True':>12} {'Conf':>10} {'OK?':>6}")
print("-" * 62)
for fnum, fname in ZD_FAMS.items():
    sub = test_zd2[test_zd2['family'].astype(str) == fnum]
    if len(sub) == 0:
        print(f"  {fname}: NO SAMPLES")
        continue
    fam_accs[fname] = sub['_ok'].mean()
    for i, (_, row) in enumerate(sub.head(5).iterrows()):
        t = ZD_CLASS_NAMES[int(row['is_ransomware'])]
        p = ZD_CLASS_NAMES[int(row['_pred'])]
        ok = "YES" if row['_ok'] else "NO"
        print(f"  {fname:<14} {i+1:>4} {p:>12} {t:>12} {row['_conf']:>9.1%} {ok:>6}")

print(f"\n[INFO] Per-family accuracy (zero-day ransomware only):")
worst_fam, worst_acc = None, 1.0
for fname, facc in fam_accs.items():
    fnum = [k for k,v in ZD_FAMS.items() if v==fname][0]
    n = (test_zd2['family'].astype(str) == fnum).sum()
    flag = " <-- WORST" if facc == min(fam_accs.values()) else ""
    print(f"  {fname:<16}: {facc:.2%}  (n={n}){flag}")
    if facc < worst_acc:
        worst_acc, worst_fam = facc, fname

if worst_fam:
    wnum = [k for k,v in ZD_FAMS.items() if v==worst_fam][0]
    wsub = test_zd2[test_zd2['family'].astype(str) == wnum]
    print(f"\n[INFO] All {len(wsub)} samples from worst family '{worst_fam}':")
    for i, (_, row) in enumerate(wsub.iterrows()):
        t = ZD_CLASS_NAMES[int(row['is_ransomware'])]
        p = ZD_CLASS_NAMES[int(row['_pred'])]
        print(f"  row {i+1}: True={t}  Pred={p}  Conf={row['_conf']:.1%}")

# ─── AUDIT 2F ────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("AUDIT 2F — Family Classifier (12-class) Evaluation")
print("-" * 70)

print(f"\n[INFO] Loading family test data ({FAM_TEST})...")
test_fam = pd.read_csv(FAM_TEST).fillna('')
true_fam = test_fam['family'].astype(int).tolist()
print(f"[INFO] {len(test_fam)} samples")

print(f"[INFO] Loading family model (498MB)...")
t2 = time.time()
fam_model = Classifier(hidden_size=768, num_classes=12)
sd2 = torch.load(FAM_MODEL, map_location='cpu')
fam_model.load_state_dict(sd2)
fam_model.eval()
print(f"[INFO] Family model loaded in {time.time()-t2:.1f}s")

print(f"[INFO] Running inference on {len(test_fam)} family samples...")
preds_fam, confs_fam, _, true_fam2 = run_inference(
    fam_model, test_fam, FAM_OUT, batch_size=8, label_col='family'
)
del fam_model, sd2; gc.collect()

fam_acc = accuracy_score(true_fam, preds_fam)
present = sorted(set(true_fam))
present_names = [FAMILY_NAMES[c] for c in present]
fam_rpt = classification_report(true_fam, preds_fam,
                                 labels=present, target_names=present_names,
                                 digits=4, zero_division=0)
print(f"\nFamily Classifier Accuracy: {fam_acc:.4f} ({fam_acc*100:.2f}%)")
print(f"\n{fam_rpt}")

for cid, cname in [(8, 'PGPCODER'), (10, 'TeslaCrypt')]:
    idxs = [i for i,l in enumerate(true_fam) if l == cid]
    if not idxs:
        print(f"  {cname}: 0 samples in test")
        continue
    correct = sum(1 for i in idxs if preds_fam[i] == cid)
    total = len(idxs)
    wrong = [FAMILY_NAMES.get(preds_fam[i], str(preds_fam[i])) for i in idxs if preds_fam[i]!=cid]
    from collections import Counter
    print(f"  {cname} (class {cid}): {correct}/{total} correct ({100.*correct/total:.1f}%)")
    if wrong: print(f"    Misclassified as: {Counter(wrong).most_common(3)}")

print("\n" + "=" * 70)
print("AUDIT 2 COMPLETE")
print("=" * 70)
