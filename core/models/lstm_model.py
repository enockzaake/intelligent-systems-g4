import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
)
from pathlib import Path


class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, task="regression"):
        super(LSTMNetwork, self).__init__()
        
        self.task = task
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        
        if task == "classification":
            self.fc2 = nn.Linear(32, 1)
            self.sigmoid = nn.Sigmoid()
        else:
            self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        x = self.dropout(last_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        if self.task == "classification":
            x = self.sigmoid(x)
        
        return x.squeeze()


class LSTMModel:
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, 
                 task="regression", learning_rate=0.001, device=None):
        
        self.task = task
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            task=task
        ).to(self.device)
        
        if task == "classification":
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.MSELoss()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.trained = False
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=64, verbose=True):
        
        train_dataset = SequenceDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            val_dataset = SequenceDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            for sequences, targets in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            if X_val is not None and y_val is not None:
                val_loss = self._validate(val_loader)
                val_losses.append(val_loss)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        self.trained = True
        return {"train_losses": train_losses, "val_losses": val_losses}
    
    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def predict(self, X, batch_size=64):
        self.model.eval()
        
        dataset = SequenceDataset(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        predictions = []
        
        with torch.no_grad():
            for sequences, _ in loader:
                sequences = sequences.to(self.device)
                outputs = self.model(sequences)
                predictions.append(outputs.cpu().numpy())
        
        predictions = np.concatenate(predictions)
        
        if self.task == "classification":
            return (predictions > 0.5).astype(int)
        
        return predictions
    
    def predict_proba(self, X, batch_size=64):
        if self.task != "classification":
            return None
        
        self.model.eval()
        
        dataset = SequenceDataset(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        probabilities = []
        
        with torch.no_grad():
            for sequences, _ in loader:
                sequences = sequences.to(self.device)
                outputs = self.model(sequences)
                probabilities.append(outputs.cpu().numpy())
        
        return np.concatenate(probabilities)
    
    def evaluate(self, X_test, y_test, batch_size=64):
        predictions = self.predict(X_test, batch_size)
        
        if self.task == "classification":
            probabilities = self.predict_proba(X_test, batch_size)
            
            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1_score": f1_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities)
            }
        else:
            mse = mean_squared_error(y_test, predictions)
            metrics = {
                "mae": mean_absolute_error(y_test, predictions),
                "mse": mse,
                "rmse": np.sqrt(mse),
                "r2": r2_score(y_test, predictions)
            }
        
        return metrics
    
    def save(self, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "task": self.task,
            "learning_rate": self.learning_rate
        }, save_path)
    
    def load(self, load_path):
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.input_size = checkpoint["input_size"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]
        self.task = checkpoint["task"]
        self.learning_rate = checkpoint["learning_rate"]
        
        self.model = LSTMNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            task=self.task
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.trained = True

