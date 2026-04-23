

# # # ransomware_0_day_detection.py

# ransomware_0_day_detection.py
# Fixed: correct label mapping for numeric family codes

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score
import argparse
import os

# ==============================================
# Custom Dataset class
# ==============================================
class Dataset(TorchDataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Combine all semantic features
        self.texts = (
            self.df['apiFeatures'].fillna('') + " " +
            self.df['dropFeatures'].fillna('') + " " +
            self.df['regFeatures'].fillna('') + " " +
            self.df['filesFeatures'].fillna('') + " " +
            self.df['filesEXTFeatures'].fillna('') + " " +
            self.df['dirFeatures'].fillna('') + " " +
            self.df['strFeatures'].fillna('')
        ).str.strip().tolist()

        # FIXED LABEL MAPPING: family 0 = Goodware, others = Ransomware
        # (from your check_leakage.py: Goodware is coded as 0)
        self.labels = (self.df['family'].astype(str) != '0').astype(int).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=1024,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ==============================================
# Classifier
# ==============================================
class Classifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2, max_seq_len=1024,
                 gpt_model_name="zhouce/RDC-GPT", compression_ratio=128):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(gpt_model_name)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.compression_ratio = compression_ratio

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.linear(pooled)
        return logits

# ==============================================
# Training
# ==============================================
def train(model, train_data, test_data, learning_rate=1e-5, epochs=20):
    train_dataset = Dataset(train_data)
    test_dataset = Dataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
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
        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")

        # Test
        model.eval()
        preds, trues = [], []
        total_acc_test = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                pred = outputs.argmax(dim=1)
                preds.extend(pred.cpu().tolist())
                trues.extend(labels.cpu().tolist())

                total_acc_test += (pred == labels).sum().item()

        test_acc = total_acc_test / len(test_data)
        print(f"Test Accuracy: {test_acc:.4f}")
        try:
            print(classification_report(trues, preds, target_names=['Goodware', 'Ransomware']))
        except ValueError as e:
            print("Classification report failed:", e)
            print(f"Unique true labels: {set(trues)}")
            print(f"Unique predicted labels: {set(preds)}")

        with open('result.txt', 'a') as f:
            f.write(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}\n")
            f.write(f"Test Accuracy: {test_acc:.4f}\n\n")

# ==============================================
# Args
# ==============================================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Data_Train_path', default='train.csv')
    parser.add_argument('--Data_Test_path', default='test.csv')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train_df = pd.read_csv(args.Data_Train_path)
    test_df = pd.read_csv(args.Data_Test_path)

    model = Classifier()
    train(model, train_df, test_df, epochs=20)































































# # ransomware_0_day_detection.py

# from Dataset import Dataset
# from LSTM.Model import Classifier
# from sklearn.model_selection import KFold
# from sklearn.metrics import f1_score, recall_score
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import argparse
# import csv
# from torch import nn 
# from torch.optim import Adam
# from transformers import GPT2Model, GPT2Tokenizer
# from tqdm import tqdm 
# from sklearn.model_selection import StratifiedKFold

# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import classification_report
# import random
# from prettytable import PrettyTable

# def main():
#     args = get_args()
#     df_train = pd.read_csv(args.Data_Train_path)
#     df_train.fillna('', inplace=True)

#     df_test = pd.read_csv(args.Data_Test_path)
#     df_test.fillna('', inplace=True)

#     labels = {
#         "Goodware": 0,
#         "Critroni": 1,
#         "CryptLocker": 2,
#         "CryptoWall": 3,
#         "KOLLAH": 4,
#         "Kovter": 5,
#         "Locker": 6,
#         "MATSNU": 7,
#         "PGPCODER": 8,
#         "Reveton": 9,
#         "TeslaCrypt": 10,
#         "Trojan-Ransom": 11
#     }

#     model = Classifier(hidden_size=768, num_classes=2, max_seq_len=1024, gpt_model_name="RDC-GPT", compression_ratio=128)
#     LR = 1e-5
#     EPOCHS = 3   # Changed from 20 → faster testing on CPU laptop (increase later if needed)
#     train(model, df_train, df_test, LR, EPOCHS)
    
# def train(model, train_data, test_data, learning_rate, epochs):
#     train = Dataset(train_data)
    
#     train_dataloader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)

#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
    
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=learning_rate)
    
#     if use_cuda:
#         model = model.cuda()
#         criterion = criterion.cuda()

#     for epoch_num in range(epochs):
#         total_acc_train = 0
#         total_loss_train = 0
        
#         for train_input, train_label in tqdm(train_dataloader):
#             train_label = train_label.to(device)
#             mask = train_input['attention_mask'].to(device)
#             input_id = train_input["input_ids"].to(device)
#             model.zero_grad()
#             output = model(input_id, mask)
#             batch_loss = criterion(output, train_label)
            
#             total_loss_train += batch_loss.item()
            
#             acc = (output.argmax(dim=1) == train_label).sum().item()
#             total_acc_train += acc

#             batch_loss.backward()
#             optimizer.step()
            
#         total_acc_val = 0
#         total_loss_val = 0
        
#         print(
#             f"Epoch {epoch_num + 1}/{epochs} | Train Loss: {total_loss_train/len(train_data):.3f} | "
#             f"Train Accuracy: {total_acc_train / len(train_data):.3f}"
#         )
#         with open('result.txt', 'a') as f:
#             print(
#                 f"Epoch {epoch_num + 1}/{epochs} | Train Loss: {total_loss_train/len(train_data):.3f} | "
#                 f"Train Accuracy: {total_acc_train / len(train_data):.3f}", file=f
#             )
        
#         # Validation / Test
#         true_labels, pred_labels = evaluate(model, test_data)
#         results = pd.DataFrame(classification_report(true_labels, pred_labels, output_dict=True))
#         table = PrettyTable()
#         table.field_names = [""] + list(results.columns)

#         for idx, row in results.iterrows():
#             table.add_row([idx] + [round(i, 3) if isinstance(i, float) else i for i in row.tolist()])
        
#         print(table)
#         file_name = f"epoch_{epoch_num + 1}.csv"
#         table2csv(table.get_string(), file_name)
#         with open('result.txt', 'a') as f:
#             print(table, file=f)
        
# def table2csv(string_data, table_name):
#     lines = [line.strip() for line in string_data.strip().split('\n')]
#     lines = [line[1:-1] for line in lines]  # Remove + borders

#     with open(table_name, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         for line in lines:
#             row = [cell.strip() for cell in line.split('|')]
#             writer.writerow([cell for cell in row if cell])

# def evaluate(model, test_data):
#     test = Dataset(test_data)
#     test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")

#     if use_cuda:
#         model = model.cuda()

#     predictions_labels = []
#     true_labels = []
#     total_acc_test = 0

#     with torch.no_grad():
#         for test_input, test_label in test_dataloader:
#             test_label = test_label.to(device)
#             mask = test_input['attention_mask'].to(device)
#             input_id = test_input['input_ids'].to(device)

#             output = model(input_id, mask)

#             acc = (output.argmax(dim=1) == test_label).sum().item()
#             total_acc_test += acc
            
#             true_labels += test_label.cpu().numpy().flatten().tolist()
#             predictions_labels += output.argmax(dim=1).cpu().numpy().flatten().tolist()

#     with open('result.txt', 'a') as f:
#         print(f'Test Accuracy: {total_acc_test / len(test_data):.3f}', file=f)
    
#     print(f'Test Accuracy: {total_acc_test / len(test_data):.3f}')
#     return true_labels, predictions_labels

# def get_args():
#     parser = argparse.ArgumentParser(description='Description of the parameters of the program run command')
#     parser.add_argument('--Data_Test_path', required=False, help='Test data after internal feature semantic processing csv path', 
#                         default=r'test.csv')
#     parser.add_argument('--Data_Train_path', required=False, help='Train data after internal feature semantic processing csv path', 
#                         default=r'train.csv')
#     args = parser.parse_args()
#     return args   

# if __name__ == "__main__":
#     main()

































# # from Dataset import Dataset
# # from LSTM.Model import Classifier
# # from sklearn.model_selection import KFold
# # from sklearn.metrics import f1_score, recall_score
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import torch
# # import argparse
# # import csv
# # from torch import nn 
# # from torch.optim import Adam
# # from transformers import GPT2Model, GPT2Tokenizer
# # from tqdm import tqdm 
# # from sklearn.model_selection import StratifiedKFold

# # from sklearn.metrics import confusion_matrix
# # from sklearn.metrics import ConfusionMatrixDisplay
# # from sklearn.metrics import ConfusionMatrixDisplay
# # from sklearn.metrics import classification_report
# # import random
# # from prettytable import PrettyTable

# # def main() :
# #     args = get_args()
# #     df_train = pd.read_csv(args.Data_Train_path)
# #     df_train.fillna('', inplace=True)


# #     df_test = pd.read_csv(args.Data_Test_path)
# #     df_test.fillna('', inplace=True)
# #     labels = {
# #         "Goodware": 0,
# #         "Critroni": 1,
# #         "CryptLocker": 2,
# #         "CryptoWall": 3,
# #         "KOLLAH": 4,
# #         "Kovter": 5,
# #         "Locker": 6,
# #         "MATSNU": 7,
# #         "PGPCODER": 8,
# #         "Reveton": 9,
# #         "TeslaCrypt": 10,
# #         "Trojan-Ransom": 11
# #             }



        
# #     model = Classifier(hidden_size=768, num_classes=2, max_seq_len=1024, gpt_model_name="RDC-GPT",compression_ratio=128)
# #     LR = 1e-5
# #     EPOCHS = 20  
# #     train(model, df_train, df_test, LR, EPOCHS)
    
# # def train(model, train_data, test_data, learning_rate, epochs):
# #     train = Dataset(train_data)
    
# #     train_dataloader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)

    
# #     use_cuda = torch.cuda.is_available()
# #     device = torch.device("cuda" if use_cuda else "cpu")
    
    
# #     criterion = nn.CrossEntropyLoss()

# #     optimizer = Adam(model.parameters(), lr=learning_rate)
    
# #     if use_cuda:
# #         model = model.cuda()
# #         criterion = criterion.cuda()

# #     for epoch_num in range(epochs):
# #         total_acc_train = 0
# #         total_loss_train = 0
        
# #         for train_input, train_label in tqdm(train_dataloader):
# #             train_label = train_label.to(device)
# #             mask = train_input['attention_mask'].to(device)
# #             input_id = train_input["input_ids"].to(device)
# #             model.zero_grad()
# #             output = model(input_id, mask)
# #             batch_loss = criterion(output, train_label)
            
# #             total_loss_train += batch_loss.item()
            
# #             acc = (output.argmax(dim=1)==train_label).sum().item()
# #             total_acc_train += acc

# #             batch_loss.backward()
# #             optimizer.step()
            
# #         total_acc_val = 0
# #         total_loss_val = 0
        
      
# #         print(
# #             f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train/len(train_data): .3f} \
# #             | Train Accuracy: {total_acc_train / len(train_data): .3f} ")
# #         with open('result.txt', 'a') as f:
# #             print(
# #             f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train/len(train_data): .3f} \
# #             | Train Accuracy: {total_acc_train / len(train_data): .3f}", file=f)
# #         # 验证
# #         true_labels, pred_labels = evaluate(model, test_data)
# #         results = pd.DataFrame(classification_report(true_labels,pred_labels,output_dict=True))
# #         table = PrettyTable()
# #         table.field_names = [""] + list(results.columns)

# #         for idx, row in results.iterrows():
# #             table.add_row([idx] + [round(i, 3) for i in row.tolist () if isinstance(i, float)])
# #         print(table)
# #         file_name = "epoch_{}.csv".format(epoch_num + 1)
# #         table2csv(table.get_string(),file_name)
# #         with open('result.txt', 'a') as f:
# #             print(table, file=f)
        
# # def table2csv(string_data, table_name):
# #     # Parses the text and converts it to a list
# #     lines = [line.strip() for line in string_data.strip().split('\n')]
# #     # Delete the + signs at the beginning and end
# #     lines = [line[1:-1] for line in lines]

# #     # Write data to CSV file
# #     with open(table_name, 'w', newline='') as csvfile:
# #         writer = csv.writer(csvfile)
# #         for line in lines:
# #             row = [cell.strip() for cell in line.split('|')]
# #             # Delete empty string and write to CSV file
# #             writer.writerow([cell for cell in row if cell])

# # # Evaluation
# # def evaluate(model, test_data):

# #     test = Dataset(test_data)

# #     test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

# #     use_cuda = torch.cuda.is_available()
# #     device = torch.device("cuda" if use_cuda else "cpu")

# #     if use_cuda:

# #         model = model.cuda()

         
# #     # Tracking variables
# #     predictions_labels = []
# #     true_labels = []
    
# #     total_acc_test = 0
# #     with torch.no_grad():

# #         for test_input, test_label in test_dataloader:

# #             test_label = test_label.to(device)
# #             mask = test_input['attention_mask'].to(device)
# #             input_id = test_input['input_ids'].to(device)

# #             output = model(input_id, mask)

# #             acc = (output.argmax(dim=1) == test_label).sum().item()
# #             total_acc_test += acc
            
# #             # add original labels
# #             true_labels += test_label.cpu().numpy().flatten().tolist()
# #             # get predicitons to list
# #             predictions_labels += output.argmax(dim=1).cpu().numpy().flatten().tolist()
# #     with open('result.txt', 'a') as f:
# #            print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}', file=f)
# #     print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
# #     return true_labels, predictions_labels
# # def get_args():
# #     parser =argparse.ArgumentParser(description='Description of the parameters of the program run command')
# #     parser.add_argument('--Data_Test_path', required=False, help='Test data after internal feature semantic processing csv path', 
# #                         default=r'test.csv')
# #     parser.add_argument('--Data_Train_path', required=False, help='Train data after internal feature semantic processing csv path', 
# #                         default=r'train.csv')
# #     args=parser.parse_args()
# #     return args   

# # if __name__ == "__main__":
# #     main()