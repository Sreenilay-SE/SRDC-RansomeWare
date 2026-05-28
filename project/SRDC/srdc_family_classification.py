
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm
from sklearn.metrics import classification_report, balanced_accuracy_score
import argparse

SAVE_DIR = '/content/drive/MyDrive/SRDC_Project'

FAMILY_NAMES = {
    0: 'Goodware',
    1: 'Citroni',
    2: 'CryptLocker',
    3: 'CryptoWall',
    4: 'Kollah',
    5: 'Kovter',
    6: 'Locker',
    7: 'Matsnu',
    8: 'PGPCODER',
    9: 'Reveton',
    10: 'TeslaCrypt',
    11: 'Trojan-Ransom'
}

class Dataset(TorchDataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.texts = (
            self.df['apiFeatures'].fillna('') + " " +
            self.df['dropFeatures'].fillna('') + " " +
            self.df['regFeatures'].fillna('') + " " +
            self.df['filesFeatures'].fillna('') + " " +
            self.df['filesEXTFeatures'].fillna('') + " " +
            self.df['dirFeatures'].fillna('') + " " +
            self.df['strFeatures'].fillna('')
        ).str.strip().tolist()

        self.labels = self.df['family'].astype(int).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text, truncation=True, max_length=1024,
            padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class Classifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=12):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained("zhouce/RDC-GPT")
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.linear(pooled)
        return logits

def train(model, train_data, test_data, epochs=20):
    train_dataset = Dataset(train_data)
    test_dataset = Dataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-5)

    results = []
    result_txt_path = f'{SAVE_DIR}/family_result.txt'
    target_names = [FAMILY_NAMES[i] for i in range(12)]

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
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

        train_acc = correct / total

        # --- EVAL ---
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                pred = outputs.argmax(dim=1)
                preds.extend(pred.cpu().tolist())
                trues.extend(labels.cpu().tolist())

        bal_acc = balanced_accuracy_score(trues, preds)
        report = classification_report(trues, preds,
                    target_names=target_names, digits=4, zero_division=0)

        print(f"\nEpoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print(report)

        results.append({
            'epoch': epoch+1,
            'train_loss': round(total_loss/len(train_loader), 4),
            'train_acc': round(train_acc, 4),
            'balanced_acc': round(bal_acc, 4)
        })

        # Save model to Drive after every epoch
        model_path = f'{SAVE_DIR}/srdc_family_epoch{epoch+1}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: srdc_family_epoch{epoch+1}.pth ✅")

        # Save result txt
        with open(result_txt_path, 'a') as f:
            f.write(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}\n")
            f.write(f"Balanced Accuracy: {bal_acc:.4f}\n")
            f.write(report + "\n")

        # Save results CSV
        pd.DataFrame(results).to_csv(
            f'{SAVE_DIR}/family_results.csv', index=False)

    print(f"\nAll done!")
    print(f"All files saved in Google Drive → SRDC_Project ✅")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Data_Train_path', default='train.csv')
    parser.add_argument('--Data_Test_path', default='test.csv')
    args = parser.parse_args()

    train_df = pd.read_csv(args.Data_Train_path)
    test_df = pd.read_csv(args.Data_Test_path)

    model = Classifier(num_classes=12)
    train(model, train_df, test_df, epochs=20)
