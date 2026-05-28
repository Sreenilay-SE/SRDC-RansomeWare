
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import pandas as pd
import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2Model
import time

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', 'project', 'SRDC')

BINARY_MODEL_PATH = os.path.join(PROJECT_ROOT, 'result', 'srdc_zero_day_BEST.pth')
FAMILY_MODEL_PATH = os.path.join(PROJECT_ROOT, 'result', 'srdc_family_BEST.pth')

FAMILY_NAMES = {
    0: 'Goodware',   1: 'Citroni',    2: 'CryptLocker',
    3: 'CryptoWall', 4: 'Kollah',     5: 'Kovter',
    6: 'Locker',     7: 'Matsnu',     8: 'PGPCODER',
    9: 'Reveton',   10: 'TeslaCrypt', 11: 'Trojan-Ransom'
}

# ── Families to test: (family_id, n_samples)
TEST_PLAN = [
    (3, 5),   # CryptoWall
    (2, 5),   # CryptLocker
    (9, 5),   # Reveton
    (0, 5),   # Goodware
]

class Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.gpt   = GPT2Model.from_pretrained("zhouce/RDC-GPT")
        self.linear = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        out    = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state.mean(dim=1)
        return self.linear(pooled)


def get_text(row):
    return " ".join([
        str(row['apiFeatures']),   str(row['dropFeatures']),
        str(row['regFeatures']),   str(row['filesFeatures']),
        str(row['filesEXTFeatures']), str(row['dirFeatures']),
        str(row['strFeatures'])
    ]).strip()


def predict(model, text, tokenizer, device):
    enc  = tokenizer(text, truncation=True, max_length=1024,
                     padding='max_length', return_tensors='pt')
    ids  = enc['input_ids'].to(device)
    mask = enc['attention_mask'].to(device)
    with torch.no_grad():
        logits = model(ids, mask)
    probs = torch.softmax(logits, dim=1)
    pred  = logits.argmax(dim=1).item()
    conf  = probs[0][pred].item() * 100
    return pred, conf


def run():
    print("\n" + "="*65, flush=True)
    print("   SRDC Family-Level Verification 🔬", flush=True)
    print("   Real CSV samples — 4 families — 20 total", flush=True)
    print("="*65 + "\n", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device : {device}", flush=True)

    print("[*] Loading Binary Model ...", flush=True)
    bin_model = Classifier(num_classes=2)
    bin_model.load_state_dict(torch.load(BINARY_MODEL_PATH, map_location=device))
    bin_model.to(device).eval()
    print("[✓] Binary model ready!", flush=True)

    print("[*] Loading Family Model ...", flush=True)
    fam_model = Classifier(num_classes=12)
    fam_model.load_state_dict(torch.load(FAMILY_MODEL_PATH, map_location=device))
    fam_model.to(device).eval()
    print("[✓] Family model ready!\n", flush=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load test.csv (has all 12 families)
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'splits', 'test.csv'))

    # Build sample set
    samples = []
    for fam_id, n in TEST_PLAN:
        rows = df[df['family'] == fam_id].sample(n=n, random_state=42)
        for _, row in rows.iterrows():
            samples.append((fam_id, row))

    total   = len(samples)
    results = []
    counter = 0

    for fam_id, row in samples:
        counter     += 1
        true_label   = int(row['is_ransomware'])
        fam_name     = FAMILY_NAMES[fam_id]
        text         = get_text(row)

        print(f"{'='*65}", flush=True)
        print(f"[{counter:02d}/{total}]  Family: {fam_name}  |  True label: {'RANSOMWARE' if true_label else 'GOODWARE'}", flush=True)

        api_preview = str(row['apiFeatures'])[:110].strip()
        print(f"  API  : {api_preview}...", flush=True)
        reg_preview = str(row['regFeatures'])[:80].strip()
        print(f"  REG  : {reg_preview}...", flush=True)

        print("  [*] Analysing", end="", flush=True)
        for _ in range(3):
            print(".", end="", flush=True)
            time.sleep(0.2)
        print(flush=True)

        # Binary prediction
        bin_pred, bin_conf = predict(bin_model, text, tokenizer, device)
        bin_correct        = (bin_pred == true_label)

        if bin_pred == 1:
            # Family classification
            fam_pred, fam_conf = predict(fam_model, text, tokenizer, device)
            fam_correct        = (fam_pred == fam_id)
            pred_name          = FAMILY_NAMES.get(fam_pred, 'Unknown')

            print(f"  🚨  RANSOMWARE  conf={bin_conf:.1f}%  binary={'✅' if bin_correct else '❌'}", flush=True)
            print(f"  🔬  Family: {pred_name}  conf={fam_conf:.1f}%  family={'✅' if fam_correct else '❌ (expected '+fam_name+')'}", flush=True)

            results.append({
                'n': counter, 'family': fam_name, 'is_ransom': true_label,
                'bin_pred': 'RANSOMWARE', 'bin_ok': bin_correct, 'bin_conf': bin_conf,
                'fam_pred': pred_name,   'fam_ok': fam_correct,  'fam_conf': fam_conf,
            })
        else:
            print(f"  ✅  GOODWARE    conf={bin_conf:.1f}%  binary={'✅' if bin_correct else '❌'}", flush=True)
            results.append({
                'n': counter, 'family': fam_name, 'is_ransom': true_label,
                'bin_pred': 'GOODWARE',  'bin_ok': bin_correct, 'bin_conf': bin_conf,
                'fam_pred': 'N/A',       'fam_ok': True,         'fam_conf': 0,
            })

        time.sleep(0.3)

    # ── SCORECARD ─────────────────────────────────────────────────────
    print(f"\n\n{'='*65}", flush=True)
    print("  📊  FINAL SCORECARD", flush=True)
    print(f"{'='*65}", flush=True)

    print(f"\n  {'#':<4} {'Family':<14} {'Expected':<12} {'BinPred':<12} {'Conf':>6}  {'Bin':>4}  {'FamPred':<14} {'Fam':>4}", flush=True)
    print(f"  {'-'*75}", flush=True)
    for r in results:
        expected = 'RANSOMWARE' if r['is_ransom'] else 'GOODWARE'
        fam_str  = r['fam_pred'] if r['is_ransom'] else '—'
        fam_ok   = ('✅' if r['fam_ok'] else '❌') if r['is_ransom'] else ' —'
        print(
            f"  {r['n']:<4} {r['family']:<14} {expected:<12} {r['bin_pred']:<12} "
            f"{r['bin_conf']:>5.1f}%  {'✅' if r['bin_ok'] else '❌':>4}  {fam_str:<14} {fam_ok:>4}",
            flush=True
        )

    # Aggregate stats
    bin_correct_total = sum(1 for r in results if r['bin_ok'])
    ransomware_results = [r for r in results if r['is_ransom']]
    fam_correct_total  = sum(1 for r in ransomware_results if r['fam_ok'])

    print(f"\n  {'─'*50}", flush=True)

    # Per-family breakdown
    print(f"\n  Per-Family Binary Accuracy:", flush=True)
    for fam_id, n in TEST_PLAN:
        name    = FAMILY_NAMES[fam_id]
        subset  = [r for r in results if r['family'] == name]
        correct = sum(1 for r in subset if r['bin_ok'])
        avg_conf = sum(r['bin_conf'] for r in subset) / len(subset)
        print(f"    {name:<14} : {correct}/{len(subset)}  avg confidence {avg_conf:.1f}%", flush=True)

    print(f"\n  Overall Binary Detection  : {bin_correct_total}/{total}  ({100*bin_correct_total/total:.1f}%)", flush=True)
    if ransomware_results:
        print(f"  Overall Family Classif.   : {fam_correct_total}/{len(ransomware_results)}  ({100*fam_correct_total/len(ransomware_results):.1f}%)", flush=True)

    if bin_correct_total == total and fam_correct_total == len(ransomware_results):
        print(f"\n  🎉  PERFECT — Model is production-ready!", flush=True)
    elif bin_correct_total / total >= 0.90:
        print(f"\n  ✅  STRONG — Model is reliable for production.", flush=True)
    else:
        print(f"\n  ⚠️   Review misclassified samples above.", flush=True)

    print(f"\n{'='*65}\n", flush=True)


if __name__ == "__main__":
    run()
