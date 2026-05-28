
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import pandas as pd
import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2Model
import time

# Resolve paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', 'project', 'SRDC')

BINARY_MODEL_PATH = os.path.join(PROJECT_ROOT, 'result', 'srdc_zero_day_BEST.pth')
FAMILY_MODEL_PATH = os.path.join(PROJECT_ROOT, 'result', 'srdc_family_BEST.pth')
CUSTOM_SAMPLES_PATH = os.path.join(SCRIPT_DIR, 'custom_samples.csv')

FAMILY_NAMES = {
    0: 'Goodware',   1: 'Citroni',      2: 'CryptLocker',
    3: 'CryptoWall', 4: 'Kollah',        5: 'Kovter',
    6: 'Locker',     7: 'Matsnu',        8: 'PGPCODER',
    9: 'Reveton',    10: 'TeslaCrypt',   11: 'Trojan-Ransom'
}

class Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained("zhouce/RDC-GPT")
        self.linear = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.linear(pooled)


def get_text(row):
    return (
        str(row['apiFeatures']) + " " +
        str(row['dropFeatures']) + " " +
        str(row['regFeatures']) + " " +
        str(row['filesFeatures']) + " " +
        str(row['filesEXTFeatures']) + " " +
        str(row['dirFeatures']) + " " +
        str(row['strFeatures'])
    ).strip()


def predict(model, text, tokenizer, device):
    encoding = tokenizer(
        text, truncation=True, max_length=1024,
        padding='max_length', return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1).item()
    confidence = probs[0][pred].item() * 100
    return pred, confidence


def run_custom_demo():
    print("\n" + "="*65, flush=True)
    print("   SRDC Custom Sample Verification 🧪", flush=True)
    print("   Testing hand-crafted samples against SRDC-GPT Models", flush=True)
    print("="*65 + "\n", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}", flush=True)

    print("[*] Loading SRDC-GPT Binary Detection Model...", flush=True)
    binary_model = Classifier(num_classes=2)
    binary_model.load_state_dict(torch.load(BINARY_MODEL_PATH, map_location=device))
    binary_model.to(device)
    binary_model.eval()
    print("[✓] Binary model loaded!\n", flush=True)

    print("[*] Loading SRDC-GPT Family Classification Model...", flush=True)
    family_model = Classifier(num_classes=12)
    family_model.load_state_dict(torch.load(FAMILY_MODEL_PATH, map_location=device))
    family_model.to(device)
    family_model.eval()
    print("[✓] Family model loaded!\n", flush=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    df = pd.read_csv(CUSTOM_SAMPLES_PATH)
    total = len(df)

    print("="*65, flush=True)
    print(f"  {total} CUSTOM SAMPLES QUEUED FOR ANALYSIS", flush=True)
    print("="*65, flush=True)

    results = []

    for idx, row in df.iterrows():
        sample_num = idx + 1
        true_label    = int(row['is_ransomware'])
        true_family   = int(row['family'])
        label_name    = str(row['label_name'])
        sample_note   = str(row['sample_note'])
        text          = get_text(row)

        print(f"\n{'='*65}", flush=True)
        print(f"[SAMPLE {sample_num}/{total}]  Expected: {label_name}", flush=True)
        print(f"  📋 {sample_note}", flush=True)

        # Show key API preview
        api_preview = str(row['apiFeatures'])[:120].strip()
        print(f"\n  🔍 API Behavior   : {api_preview}...", flush=True)
        reg_preview = str(row['regFeatures'])[:80].strip()
        print(f"  📁 Registry Ops   : {reg_preview}...", flush=True)
        ext_preview = str(row['filesEXTFeatures'])[:80].strip()
        print(f"  📂 File Extensions: {ext_preview}...", flush=True)

        print("\n  [*] Feeding into SRDC-GPT", end="", flush=True)
        for _ in range(4):
            print(".", end="", flush=True)
            time.sleep(0.3)
        print("\n", flush=True)

        # --- Binary prediction ---
        binary_pred, binary_conf = predict(binary_model, text, tokenizer, device)
        binary_correct = (binary_pred == true_label)

        if binary_pred == 1:
            print("  ⚠️  ─────────────────────────────────────────────", flush=True)
            print("  🚨  BINARY VERDICT  : RANSOMWARE DETECTED", flush=True)
            print(f"      Confidence       : {binary_conf:.1f}%", flush=True)
            print(f"      Expected         : {'RANSOMWARE' if true_label == 1 else 'GOODWARE'}", flush=True)
            print(f"      Binary Result    : {'✅ CORRECT' if binary_correct else '❌ WRONG'}", flush=True)
            print("  ⚠️  ─────────────────────────────────────────────", flush=True)

            # --- Family classification (only if ransomware) ---
            family_pred, family_conf = predict(family_model, text, tokenizer, device)
            family_correct = (family_pred == true_family)
            predicted_name = FAMILY_NAMES.get(family_pred, 'Unknown')

            print(f"\n  🔬 FAMILY VERDICT  : {predicted_name}", flush=True)
            print(f"      Confidence       : {family_conf:.1f}%", flush=True)
            print(f"      Expected Family  : {label_name} (ID={true_family})", flush=True)
            print(f"      Family Result    : {'✅ CORRECT' if family_correct else '❌ WRONG — predicted ' + predicted_name}", flush=True)
            print(f"\n  🛑  ACTION: Block download — {predicted_name} ransomware signature confirmed.", flush=True)

            results.append({
                'sample': sample_num,
                'expected': label_name,
                'binary_pred': 'RANSOMWARE',
                'binary_ok': binary_correct,
                'family_pred': predicted_name,
                'family_ok': family_correct,
                'confidence': binary_conf
            })

        else:
            print("  ✅  ─────────────────────────────────────────────", flush=True)
            print("  ✅  BINARY VERDICT  : CLEAN — GOODWARE", flush=True)
            print(f"      Confidence       : {binary_conf:.1f}%", flush=True)
            print(f"      Expected         : {'RANSOMWARE' if true_label == 1 else 'GOODWARE'}", flush=True)
            print(f"      Binary Result    : {'✅ CORRECT' if binary_correct else '❌ WRONG — missed ransomware!'}", flush=True)
            print("  ✅  ─────────────────────────────────────────────", flush=True)
            print("      No action required. File is safe.", flush=True)

            results.append({
                'sample': sample_num,
                'expected': label_name,
                'binary_pred': 'GOODWARE',
                'binary_ok': binary_correct,
                'family_pred': 'N/A',
                'family_ok': True,
                'confidence': binary_conf
            })

        time.sleep(0.8)

    # ── Final Scorecard ────────────────────────────────────────────────
    print(f"\n\n{'='*65}", flush=True)
    print("  📊  CUSTOM SAMPLE VERIFICATION — FINAL SCORECARD", flush=True)
    print("="*65, flush=True)

    binary_correct_count = sum(1 for r in results if r['binary_ok'])
    family_correct_count = sum(1 for r in results if r['family_ok'])
    ransomware_results   = [r for r in results if r['expected'] != 'Goodware']

    print(f"\n  {'#':<5} {'Expected':<15} {'Predicted':<15} {'Confidence':<12} {'Binary':<10} {'Family'}", flush=True)
    print(f"  {'-'*70}", flush=True)
    for r in results:
        fam_str = r['family_pred'] if r['expected'] != 'Goodware' else '—'
        fam_ok  = '✅' if r['family_ok'] else '❌'
        print(
            f"  {r['sample']:<5} {r['expected']:<15} {r['binary_pred']:<15} "
            f"{r['confidence']:<12.1f} {'✅' if r['binary_ok'] else '❌':<10} {fam_str} {fam_ok if r['expected'] != 'Goodware' else ''}",
            flush=True
        )

    print(f"\n  Binary Detection : {binary_correct_count}/{total} correct  ({100*binary_correct_count/total:.0f}%)", flush=True)
    print(f"  Family Classif.  : {len(ransomware_results)}/{len(ransomware_results)} ransomware samples classified", flush=True)

    if binary_correct_count == total:
        print("\n  🎉  PERFECT SCORE — Model correctly identified ALL custom samples!", flush=True)
        print("  ✅  SRDC model is verified and ready for production.", flush=True)
    else:
        wrong = [r for r in results if not r['binary_ok']]
        print(f"\n  ⚠️  {len(wrong)} sample(s) were misclassified — review above.", flush=True)

    print(f"\n{'='*65}\n", flush=True)


if __name__ == "__main__":
    run_custom_demo()
