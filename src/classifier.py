## Hanyun Hu ##
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
from tqdm import tqdm


class ABSADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.data = pd.read_csv(file_path, sep='\t', header=None,
                                names=['sentiment', 'aspect', 'target', 'position', 'text'])
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = f"Aspect: {row['aspect']} Target: {row['target']} Text: {row['text']}"
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label_map[row['sentiment']])
        }

class Classifier:
    def __init__(self):
        self.model_name = 'distilbert-base-uncased'
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_name, num_labels=3)
        self.device = None

    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        self.device = device
        self.model.to(self.device)

        train_dataset = ABSADataset(train_filename, self.tokenizer)
        dev_dataset = ABSADataset(dev_filename, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=16)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

        num_epochs = 5  # Increased from 3 to allow for early stopping
        patience = 3  # Number of epochs to wait before early stopping
        best_accuracy = 0
        no_improvement = 0
        best_model = None

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

            # Evaluate on dev set
            dev_accuracy = self.evaluate(dev_loader)
            print(f"Dev Accuracy: {dev_accuracy:.4f}")

            # Early stopping logic
            if dev_accuracy > best_accuracy:
                best_accuracy = dev_accuracy
                no_improvement = 0
                best_model = self.model.state_dict().copy()
            else:
                no_improvement += 1

            if no_improvement >= patience:
                print(f"Early stopping triggered. Best accuracy: {best_accuracy:.4f}")
                break

        # Load the best model
        if best_model is not None:
            self.model.load_state_dict(best_model)

    def evaluate(self, data_loader):
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)

                predictions.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())

        accuracy = accuracy_score(true_labels, predictions)
        return accuracy

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        dataset = ABSADataset(data_filename, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=16)

        predictions = []
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)

                predictions.extend([label_map[p.item()] for p in preds])

        return predictions
