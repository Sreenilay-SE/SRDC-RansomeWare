"""
AUDIT 2 — TRAINING LOG ANALYSIS (no model needed)
Extracts all statistically meaningful findings from zero_day_results.csv
and family_results.csv that are relevant to model validity checks.
Author: ML Auditor (Antigravity)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

BASE = r"C:\Users\sree nilay\Downloads\DOMAIN-PRO-SRDC\DOMAIN-PRO-SRDC\project\SRDC"

zd  = pd.read_csv(f"{BASE}\\result\\zero_day_results.csv")
fam = pd.read_csv(f"{BASE}\\result\\family_results.csv")

print("=" * 70)
print("TRAINING LOG ANALYSIS — Zero-Day Model")
print("=" * 70)

print(f"\nEpochs logged: {len(zd)}")
print(f"Train acc range: {zd['train_acc'].min():.4f} – {zd['train_acc'].max():.4f}")
print(f"Test  acc range: {zd['test_acc'].min():.4f} – {zd['test_acc'].max():.4f}")
print(f"Train/Test gap at epoch 20: {zd.iloc[-1]['train_acc'] - zd.iloc[-1]['test_acc']:+.4f}")

# Overfitting signal: train acc monotonically rises while test oscillates
test_accs = zd['test_acc'].tolist()
train_accs = zd['train_acc'].tolist()

# Variance in test_acc (a genuine converged model should have low variance)
test_std = np.std(test_accs)
print(f"\nTest accuracy std dev across 20 epochs: {test_std:.4f}  ({test_std*100:.2f}%)")
print(f"  (A stable model should have std < 1%; {test_std*100:.1f}% is {'UNSTABLE' if test_std > 0.02 else 'stable'})")

# How many epochs does test_acc actually improve?
improvements = sum(1 for i in range(1, len(test_accs)) if test_accs[i] > test_accs[i-1])
drops        = sum(1 for i in range(1, len(test_accs)) if test_accs[i] < test_accs[i-1])
print(f"\nEpoch-to-epoch test_acc changes:")
print(f"  Improvements: {improvements}/19")
print(f"  Drops:        {drops}/19")

print(f"\nBest test_acc epoch: {zd.loc[zd['test_acc'].idxmax(), 'epoch']} "
      f"({zd['test_acc'].max():.4f})")
print(f"Worst test_acc epoch: {zd.loc[zd['test_acc'].idxmin(), 'epoch']} "
      f"({zd['test_acc'].min():.4f})")
print(f"Range: {zd['test_acc'].max() - zd['test_acc'].min():.4f} "
      f"({(zd['test_acc'].max() - zd['test_acc'].min())*100:.1f}%)")

# Epoch 11 is the reported BEST — check if it's genuinely best
e11 = zd[zd['epoch'] == 11].iloc[0]
is_best_test = e11['test_acc'] == zd['test_acc'].max()
print(f"\nEpoch 11 (claimed BEST): test_acc={e11['test_acc']:.4f} train_acc={e11['train_acc']:.4f}")
print(f"  Is epoch 11 the actual best test_acc epoch? {'YES' if is_best_test else 'NO'}")

# Test for whether the high training acc is just memorisation
# Signal: train_acc > 0.99 AND test_acc varies by >5% = memorisation
final_train = train_accs[-1]
if final_train > 0.99 and (max(test_accs) - min(test_accs)) > 0.05:
    print(f"\n[SIGNAL] Memorisation pattern detected:")
    print(f"  Final train_acc={final_train:.4f} (>99%) while test_acc swings "
          f"{(max(test_accs)-min(test_accs))*100:.1f}% across epochs")
    print(f"  This is characteristic of a model memorising training samples, "
          f"not genuinely generalising.")

print("\n" + "=" * 70)
print("TRAINING LOG ANALYSIS — Family Classifier")
print("=" * 70)

print(f"\nEpochs logged: {len(fam)}")
print(f"Train acc range:    {fam['train_acc'].min():.4f} – {fam['train_acc'].max():.4f}")
print(f"Balanced acc range: {fam['balanced_acc'].min():.4f} – {fam['balanced_acc'].max():.4f}")

bal_accs = fam['balanced_acc'].tolist()
print(f"\nBest balanced_acc: epoch {fam.loc[fam['balanced_acc'].idxmax(),'epoch']} "
      f"({fam['balanced_acc'].max():.4f} = {fam['balanced_acc'].max()*100:.1f}%)")
print(f"Chance level (12 classes): {1/12:.4f} = {1/12*100:.1f}%")
print(f"Model vs chance: {(fam['balanced_acc'].max() - 1/12)*100:+.1f}% above chance")

print(f"\nNote: balanced_acc peaked at {fam['balanced_acc'].max():.4f} at epoch 16, "
      f"then dropped to {bal_accs[-1]:.4f} at epoch 20")
print(f"  This means the epoch saved as BEST may not be epoch 16.")
print(f"  Family classifier shows clear non-convergence in test metric.")

# Compute the train/balanced gap: proxy for overfitting
final_train_fam = fam.iloc[-1]['train_acc']
final_bal_fam   = fam.iloc[-1]['balanced_acc']
print(f"\nEpoch 20 train_acc={final_train_fam:.4f} vs balanced_acc={final_bal_fam:.4f}")
print(f"  Gap = {(final_train_fam - final_bal_fam)*100:.1f}% — "
      f"{'severe overfitting' if final_train_fam - final_bal_fam > 0.4 else 'moderate gap'}")
