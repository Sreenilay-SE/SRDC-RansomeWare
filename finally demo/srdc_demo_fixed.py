
import sys
import os
sys.stdout.reconfigure(line_buffering=True)  # Fix Colab output buffering

import pandas as pd
import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2Model
import time
import random

BINARY_MODEL_PATH = '/content/drive/MyDrive/SRDC_Project/srdc_zero_day_BEST.pth'
FAMILY_MODEL_PATH = '/content/drive/MyDrive/SRDC_Project/srdc_family_BEST.pth'

FAMILY_NAMES = {
    0: 'Goodware', 1: 'Citroni', 2: 'CryptLocker',
    3: 'CryptoWall', 4: 'Kollah', 5: 'Kovter',
    6: 'Locker', 7: 'Matsnu', 8: 'PGPCODER',
    9: 'Reveton', 10: 'TeslaCrypt', 11: 'Trojan-Ransom'
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

def slow_print(text, delay=0.03):
    print(text, flush=True)

def run_demo():
    print("\n" + "="*60, flush=True)
    print("   SRDC Ransomware Detection System 🛡️", flush=True)
    print("   Powered by GPT-2 Semantic Analysis", flush=True)
    print("="*60 + "\n", flush=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}", flush=True)
    
    print("[*] Loading SRDC-GPT Binary Detection Model...", flush=True)
    binary_model = Classifier(num_classes=2)
    binary_model.load_state_dict(torch.load(BINARY_MODEL_PATH, map_location=device))
    binary_model.to(device)
    binary_model.eval()
    print("[✓] Binary model loaded!", flush=True)

    print("[*] Loading SRDC-GPT Family Classification Model...", flush=True)
    family_model = Classifier(num_classes=12)
    family_model.load_state_dict(torch.load(FAMILY_MODEL_PATH, map_location=device))
    family_model.to(device)
    family_model.eval()
    print("[✓] Family model loaded!", flush=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    df_binary = pd.read_csv('zero_day_test.csv')
    df_family = pd.read_csv('test.csv')

    ransomware_samples = df_binary[df_binary['is_ransomware'] == 1].sample(2, random_state=42)
    goodware_samples   = df_binary[df_binary['is_ransomware'] == 0].sample(1, random_state=42)
    demo_samples = pd.concat([ransomware_samples, goodware_samples]).sample(frac=1, random_state=7).reset_index(drop=True)

    print("="*60)
    print(" SANDBOX MONITORING ACTIVE — 3 SAMPLES QUEUED", flush=True)
    print("="*60, flush=True)

    for i, row in demo_samples.iterrows():
        sample_num = i + 1
        true_label = int(row['is_ransomware'])
        text = get_text(row)

        print(f"\n{'='*60}")
        print(f"[*] Sample {sample_num}/3 entering sandbox...", flush=True)

        api_preview = str(row['apiFeatures'])[:120].strip()
        print(f"\n[*] Captured API behavior (preview):", flush=True)
        print(f"    → {api_preview}...", flush=True)

        reg_preview = str(row['regFeatures'])[:80].strip()
        print(f"[*] Registry activity:", flush=True)
        print(f"    → {reg_preview}...", flush=True)

        file_preview = str(row['filesEXTFeatures'])[:80].strip()
        print(f"[*] File extensions accessed:", flush=True)
        print(f"    → {file_preview}...", flush=True)

        print("\n[*] Feeding behavior into SRDC-GPT model...", flush=True)
        print("[*] Running semantic analysis", end="", flush=True)
        for _ in range(3):
            print(".", end="", flush=True)
            time.sleep(0.4)
        print("\n", flush=True)

        binary_pred, binary_conf = predict(binary_model, text, tokenizer, device)

        print()
        if binary_pred == 1:
            print("⚠️  ══════════════════════════════════════", flush=True)
            print("🚨  RANSOMWARE DETECTED!", flush=True)
            print(f"    Confidence : {binary_conf:.1f}%", flush=True)
            print(f"    True Label : {'RANSOMWARE' if true_label==1 else 'GOODWARE'}", flush=True)
            print(f"    Result     : {'✅ CORRECT' if binary_pred==true_label else '❌ WRONG'}", flush=True)
            print("⚠️  ══════════════════════════════════════", flush=True)

            family_pred, family_conf = predict(family_model, text, tokenizer, device)
            true_family = FAMILY_NAMES.get(int(row['family']), 'Unknown')
            
            print("\n[*] Running Family Classification...", flush=True)
            print(f"🔍  Family Identified : {FAMILY_NAMES[family_pred]}", flush=True)
            print(f"    Confidence        : {family_conf:.1f}%", flush=True)
            print(f"    True Family       : {true_family}", flush=True)
            print(f"    Result            : {'✅ CORRECT' if family_pred==row['family'] else '❌ WRONG'}", flush=True)
            print()
            print("🛑  ACTION: ISOLATE SYSTEM IMMEDIATELY", flush=True)
            print(f"    Threat: {FAMILY_NAMES[family_pred]} ransomware confirmed.", flush=True)

        else:
            print("✅  ══════════════════════════════════════", flush=True)
            print("✅  SYSTEM IS CLEAN — GOODWARE", flush=True)
            print(f"    Confidence : {binary_conf:.1f}%", flush=True)
            print(f"    True Label : {'RANSOMWARE' if true_label==1 else 'GOODWARE'}", flush=True)
            print(f"    Result     : {'✅ CORRECT' if binary_pred==true_label else '❌ WRONG'}", flush=True)
            print("✅  ══════════════════════════════════════", flush=True)
            print("    No action required. Sample is safe.", flush=True)

        time.sleep(1)

    print(f"\n{'='*60}")
    print("  SANDBOX ANALYSIS COMPLETE", flush=True)
    print("  All 3 samples processed by SRDC system.", flush=True)
    print("="*60 + "\n", flush=True)

if __name__ == "__main__":
    run_demo()
