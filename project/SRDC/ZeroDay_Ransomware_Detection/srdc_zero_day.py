
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import argparse

SAVE_DIR = '/content/drive/MyDrive/SRDC_Project'

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

        self.labels = self.df['is_ransomware'].astype(int).tolist()

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
    def __init__(self, hidden_size=768, num_classes=2):
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
    result_txt_path = f'{SAVE_DIR}/zero_day_result.txt'

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

        test_acc = accuracy_score(trues, preds)
        report = classification_report(trues, preds,
                    target_names=['Goodware', 'Ransomware'], digits=4)

        print(f"\nEpoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(report)

        results.append({
            'epoch': epoch+1,
            'train_loss': round(total_loss/len(train_loader), 4),
            'train_acc': round(train_acc, 4),
            'test_acc': round(test_acc, 4)
        })

        # ✅ Save model to Drive after EVERY epoch
        model_path = f'{SAVE_DIR}/srdc_zero_day_epoch{epoch+1}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: srdc_zero_day_epoch{epoch+1}.pth ✅")

        # ✅ Save result txt to Drive after every epoch
        with open(result_txt_path, 'a') as f:
            f.write(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}\n")
            f.write(f"Test Accuracy: {test_acc:.4f}\n")
            f.write(report + "\n")

        # ✅ Save results CSV to Drive after every epoch
        pd.DataFrame(results).to_csv(f'{SAVE_DIR}/zero_day_results.csv', index=False)

    print(f"\nAll done! Final model: srdc_zero_day_epoch20.pth")
    print(f"All files saved in Google Drive → SRDC_Project folder ✅")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Data_Train_path', default='zero_day_train.csv')
    parser.add_argument('--Data_Test_path', default='zero_day_test.csv')
    args = parser.parse_args()

    train_df = pd.read_csv(args.Data_Train_path)
    test_df = pd.read_csv(args.Data_Test_path)

    model = Classifier()
    train(model, train_df, test_df, epochs=20)
