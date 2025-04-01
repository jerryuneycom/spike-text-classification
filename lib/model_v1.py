import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from transformers import XLMRobertaTokenizer
from lib.model import IModel
from torch.utils.data import Dataset

class ScamSMSDatasetV1(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.data = []
        
        for _, d in enumerate(data):
            token_ids = self.tokenizer.encode(d['text'])
            chunk = [0] * max_length
            chunk[:len(token_ids)] = token_ids[:max_length]
            self.data.append({
                'text': torch.tensor(chunk),
                'label': d['label']
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data['text'], data['label']
    
class ScamSMSDatasetV11(Dataset):
    def __init__(self, data, tokenizer, max_length=128, stride=64):
        self.tokenizer = tokenizer
        self.data = []
        
        for _, d in enumerate(data):
            token_ids = self.tokenizer.encode(d['text'])
            for i in range(0, len(token_ids), stride):
                chunk = [0] * max_length
                chunk[:len(token_ids[i:i+max_length])] = token_ids[i:i+max_length]
                self.data.append({
                    'text': torch.tensor(chunk),
                    'label': d['label']
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data['text'], data['label']

class ScamSMSClassifierV1(IModel):
    def __init__(self, _, embed_dim, hidden_dim, num_classes):
        super(ScamSMSClassifierV1, self).__init__()

        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        vocab_size = self.tokenizer.vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dim, num_classes)

        # For example
        print(f"Tokenizer loaded with size {self.tokenizer.vocab_size}")

    def forward(self, x):
        x = self.embedding(x).mean(dim=1)  # Average embeddings
        x = self.fc(x)
        x = self.relu(x)
        x = self.output(x)
        return x

    def encode(self, text):
        return self.tokenizer.encode(text)
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
    

    def create_data_set(self, data):
        return ScamSMSDatasetV11(data, self.tokenizer)
    
    def start_train(self, data, lr=0.0001, epochs=10):
        train_data_set = self.create_data_set(data)
        train_loader = DataLoader(train_data_set, batch_size=1)

        print("Training...")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()
        for i in range(epochs):
            for texts, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print(f"Epoch: {i+1}/{epochs} completed")

    def start_eval(self, data):
        self.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for d in data:
                texts, labels = d['text'], d['label']
                token_ids = self.tokenizer.encode(texts)
                outputs = self(torch.tensor(token_ids).unsqueeze(0).to(self.device))
                total += 1
                _, predicted = torch.max(outputs, 1)
                print(f"With test {texts} predicted: {predicted} actual: {labels}")
                correct += (predicted == labels).sum().item()

        print(f"Accuracy: {correct / total  * 100}%")

        