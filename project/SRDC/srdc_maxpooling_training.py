"""
SRDC MaxPooling Combinatorial Model — Kaggle Training Script
=============================================================
This script implements the CORRECT hierarchical MaxPooling architecture
from the SRDC research paper (AAAI-2025).

Key corrections over the original broken code:
  1. Fixed pooling dimension: transpose before AdaptiveMaxPool1d so we
     compress the SEQUENCE dimension (tokens), not the hidden dimension.
  2. Uses clean, non-leaking dataset splits from clean_splits/.
  3. Automated best-checkpoint saving based on validation accuracy.
  4. Gradient checkpointing enabled to fit 7x GPT-2 passes on T4 GPU.

Usage on Kaggle:
  1. Upload this script + the 4 CSV files from clean_splits/ as a dataset.
  2. Select GPU T4 x2 accelerator.
  3. Run in a notebook cell:
       !python srdc_maxpooling_training.py --task zero_day
       !python srdc_maxpooling_training.py --task family
  4. Download the _BEST.pth files from /kaggle/working/

Author: SRDC Shield Team
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score

# ======================================================================
# Configuration
# ======================================================================
FEATURE_COLUMNS = [
    'apiFeatures', 'dropFeatures', 'regFeatures',
    'filesFeatures', 'filesEXTFeatures', 'dirFeatures', 'strFeatures'
]
NUM_FEATURES = len(FEATURE_COLUMNS)  # 7

MAX_SEQ_LEN = 1024
HIDDEN_SIZE = 768
COMPRESSION_RATIO = 64  # Pool each feature's sequence down to 64 tokens
GPT_MODEL_NAME = "zhouce/RDC-GPT"
LEARNING_RATE = 1e-5
EPOCHS = 20
BATCH_SIZE = 1  # Must be 1 for 7x GPT-2 passes to fit in T4 VRAM

FAMILY_NAMES = {
    0: 'Goodware', 1: 'Citroni', 2: 'CryptLocker',
    3: 'CryptoWall', 4: 'Kollah', 5: 'Kovter',
    6: 'Locker', 7: 'Matsnu', 8: 'PGPCODER',
    9: 'Reveton', 10: 'TeslaCrypt', 11: 'Trojan-Ransom'
}


# ======================================================================
# Dataset: Tokenizes each of the 7 features INDEPENDENTLY
# ======================================================================
class SRDCDataset(TorchDataset):
    """
    Each sample produces 7 separate tokenized sequences (one per feature),
    stacked into tensors of shape (7, max_seq_len).
    This is the CORRECT approach matching the paper's Dataset.py.
    """

    def __init__(self, dataframe, task='zero_day'):
        self.df = dataframe.reset_index(drop=True).fillna('')
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Pre-extract the 7 feature text columns
        self.feature_texts = []
        for col in FEATURE_COLUMNS:
            self.feature_texts.append(self.df[col].astype(str).tolist())

        # Labels
        if task == 'zero_day':
            # Binary: 0 = Goodware, 1 = Ransomware
            self.labels = (self.df['family'].astype(int) != 0).astype(int).tolist()
        else:
            # Multi-class: 12 family classes
            self.labels = self.df['family'].astype(int).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Tokenize each feature column separately
        all_input_ids = []
        all_attention_masks = []

        for feat_idx in range(NUM_FEATURES):
            text = self.feature_texts[feat_idx][idx]
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                padding='max_length',
                return_tensors='pt'
            )
            all_input_ids.append(encoding['input_ids'].squeeze(0))        # (max_seq_len,)
            all_attention_masks.append(encoding['attention_mask'].squeeze(0))  # (max_seq_len,)

        # Stack into (7, max_seq_len)
        input_ids = torch.stack(all_input_ids, dim=0)
        attention_mask = torch.stack(all_attention_masks, dim=0)

        return {
            'input_ids': input_ids,           # (7, 1024)
            'attention_mask': attention_mask,  # (7, 1024)
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ======================================================================
# Model: Corrected MaxPooling Combinatorial Classifier
# ======================================================================
class MaxPoolingClassifier(nn.Module):
    """
    Corrected implementation of the SRDC paper's combinatorial model.

    For each of the 7 feature channels:
      1. Run GPT-2 → output shape (batch, seq_len, hidden_size)
      2. Transpose to (batch, hidden_size, seq_len)
      3. AdaptiveMaxPool1d → (batch, hidden_size, compression_ratio)
      4. Transpose back to (batch, compression_ratio, hidden_size)

    Concatenate all 7 pooled outputs along dim=1:
      → (batch, 7 * compression_ratio, hidden_size)

    Flatten and classify:
      → fc1 input = 7 * compression_ratio * hidden_size
    """

    def __init__(self, hidden_size=HIDDEN_SIZE, num_classes=2,
                 compression_ratio=COMPRESSION_RATIO):
        super().__init__()
        self.gpt2model = GPT2Model.from_pretrained(GPT_MODEL_NAME)
        # Enable gradient checkpointing to save GPU VRAM
        self.gpt2model.gradient_checkpointing_enable()

        # Pool the SEQUENCE dimension down to compression_ratio
        self.pooling = nn.AdaptiveMaxPool1d(compression_ratio)

        # Final classifier: input is 7 channels × compression_ratio × hidden_size
        fc_input_dim = NUM_FEATURES * compression_ratio * hidden_size
        self.fc1 = nn.Linear(fc_input_dim, num_classes)

        self.compression_ratio = compression_ratio

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids:      (batch, 7, seq_len)
            attention_mask:  (batch, 7, seq_len)
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = input_ids.shape[0]

        # Split the 7 features
        # input_ids[:, i, :] → (batch, seq_len)
        pooled_outputs = []

        for i in range(NUM_FEATURES):
            sub_input_ids = input_ids[:, i, :]       # (batch, seq_len)
            sub_mask = attention_mask[:, i, :]         # (batch, seq_len)

            # GPT-2 forward pass
            gpt_out = self.gpt2model(
                input_ids=sub_input_ids,
                attention_mask=sub_mask
            ).last_hidden_state  # (batch, seq_len, hidden_size)

            # === THE FIX: Transpose before pooling ===
            # From (batch, seq_len, hidden_size) to (batch, hidden_size, seq_len)
            gpt_out_t = gpt_out.transpose(1, 2)

            # Pool the sequence dimension: (batch, hidden_size, seq_len) → (batch, hidden_size, compression_ratio)
            pooled = self.pooling(gpt_out_t)

            # Transpose back: (batch, compression_ratio, hidden_size)
            pooled = pooled.transpose(1, 2)

            pooled_outputs.append(pooled)

        # Concatenate all 7 channels: (batch, 7 * compression_ratio, hidden_size)
        result = torch.cat(pooled_outputs, dim=1)

        # Flatten and classify
        result_flat = result.view(batch_size, -1)  # (batch, 7 * compression_ratio * hidden_size)
        logits = self.fc1(result_flat)

        return logits


# ======================================================================
# Training Loop with Automated Best Checkpointing
# ======================================================================
def train_model(task, train_csv, test_csv, save_dir):
    print("=" * 70)
    print(f"  SRDC MaxPooling Training — Task: {task.upper()}")
    print("=" * 70)

    # Determine number of classes
    if task == 'zero_day':
        num_classes = 2
        target_names = ['Goodware', 'Ransomware']
        metric_name = 'accuracy'
    else:
        num_classes = 12
        target_names = [FAMILY_NAMES[i] for i in range(12)]
        metric_name = 'balanced_accuracy'

    # Load data
    print(f"[INFO] Loading training data from {train_csv}...")
    train_df = pd.read_csv(train_csv)
    print(f"[INFO] Loading test data from {test_csv}...")
    test_df = pd.read_csv(test_csv)
    print(f"[INFO] Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Create datasets and loaders
    print("[INFO] Tokenizing datasets (this takes a few minutes)...")
    train_dataset = SRDCDataset(train_df, task=task)
    test_dataset = SRDCDataset(test_df, task=task)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = MaxPoolingClassifier(
        hidden_size=HIDDEN_SIZE,
        num_classes=num_classes,
        compression_ratio=COMPRESSION_RATIO
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Results tracking
    results = []
    best_metric = 0.0
    best_epoch = -1
    os.makedirs(save_dir, exist_ok=True)

    if task == 'zero_day':
        best_model_name = 'srdc_zero_day_BEST.pth'
    else:
        best_model_name = 'srdc_family_BEST.pth'

    result_log_path = os.path.join(save_dir, f'{task}_training_log.txt')

    print("-" * 70)
    print(f"[INFO] Starting training for {EPOCHS} epochs...")
    print("-" * 70)

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # ---- TRAIN ----
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)           # (batch, 7, 1024)
            attention_mask = batch['attention_mask'].to(device)  # (batch, 7, 1024)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # ---- EVAL ----
        model.eval()
        preds_all, trues_all = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [EVAL]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                pred = outputs.argmax(dim=1)
                preds_all.extend(pred.cpu().tolist())
                trues_all.extend(labels.cpu().tolist())

        # Calculate metrics
        if task == 'zero_day':
            current_metric = accuracy_score(trues_all, preds_all)
        else:
            current_metric = balanced_accuracy_score(trues_all, preds_all)

        report = classification_report(
            trues_all, preds_all,
            target_names=target_names,
            digits=4, zero_division=0
        )

        epoch_time = time.time() - epoch_start

        # Print results
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.0f}s")
        print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Test {metric_name}: {current_metric:.4f}")
        if current_metric > best_metric:
            print(f"  ⭐ NEW BEST! ({best_metric:.4f} → {current_metric:.4f})")
        print(f"{'='*60}")
        print(report)

        # Save results
        results.append({
            'epoch': epoch + 1,
            'train_loss': round(avg_loss, 4),
            'train_acc': round(train_acc, 4),
            f'test_{metric_name}': round(current_metric, 4),
            'time_seconds': round(epoch_time, 1)
        })

        # Log to file
        with open(result_log_path, 'a') as f:
            f.write(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Test {metric_name}: {current_metric:.4f} | "
                    f"Time: {epoch_time:.0f}s\n")
            f.write(report + "\n")

        # Save CSV results
        pd.DataFrame(results).to_csv(
            os.path.join(save_dir, f'{task}_results.csv'), index=False
        )

        # ---- AUTOMATED BEST CHECKPOINT ----
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            best_path = os.path.join(save_dir, best_model_name)
            torch.save(model.state_dict(), best_path)
            print(f"✅ Best model saved: {best_model_name} "
                  f"(epoch {best_epoch}, {metric_name}={best_metric:.4f})")

        # Also save every epoch checkpoint (optional, for analysis)
        epoch_path = os.path.join(save_dir, f'srdc_{task}_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), epoch_path)

    print("\n" + "=" * 70)
    print(f"  TRAINING COMPLETE!")
    print(f"  Best {metric_name}: {best_metric:.4f} at epoch {best_epoch}")
    print(f"  Best model saved as: {best_model_name}")
    print(f"  All results saved to: {save_dir}/")
    print("=" * 70)

    return best_metric, best_epoch


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SRDC MaxPooling Combinatorial Model Training'
    )
    parser.add_argument(
        '--task', type=str, required=True,
        choices=['zero_day', 'family'],
        help='Training task: "zero_day" (binary) or "family" (12-class)'
    )
    parser.add_argument(
        '--data_dir', type=str, default='./clean_splits',
        help='Directory containing the clean CSV splits'
    )
    parser.add_argument(
        '--save_dir', type=str, default='./maxpooling_results',
        help='Directory to save trained models and results'
    )
    parser.add_argument(
        '--epochs', type=int, default=20,
        help='Number of training epochs'
    )
    args = parser.parse_args()

    EPOCHS = args.epochs

    if args.task == 'zero_day':
        train_csv = os.path.join(args.data_dir, 'zero_day_train.csv')
        test_csv = os.path.join(args.data_dir, 'zero_day_test.csv')
    else:
        train_csv = os.path.join(args.data_dir, 'train.csv')
        test_csv = os.path.join(args.data_dir, 'test.csv')

    train_model(
        task=args.task,
        train_csv=train_csv,
        test_csv=test_csv,
        save_dir=args.save_dir
    )
