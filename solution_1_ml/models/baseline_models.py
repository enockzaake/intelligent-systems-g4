from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
import numpy as np
import pickle
from pathlib import Path


class LogisticRegressionModel:
    def __init__(self, max_iter=1000, random_state=42, class_weight='balanced'):
        self.model = LogisticRegression(
            max_iter=max_iter, 
            random_state=random_state,
            class_weight=class_weight
        )
        self.trained = False
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.trained = True
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics
    
    def get_feature_importance(self, feature_names):
        if not self.trained:
            return None
        
        coefficients = self.model.coef_[0]
        importance_dict = {
            name: abs(coef) for name, coef in zip(feature_names, coefficients)
        }
        sorted_importance = sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_importance
    
    def save(self, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "wb") as f:
            pickle.dump(self.model, f)
    
    def load(self, load_path):
        with open(load_path, "rb") as f:
            self.model = pickle.load(f)
        self.trained = True


class RandomForestModel:
    def __init__(self, task="regression", n_estimators=100, max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1, class_weight='balanced', random_state=42):
        self.task = task
        
        if task == "classification":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                class_weight=class_weight,
                n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1
            )
        
        self.trained = False
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.trained = True
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.task == "classification":
            return self.model.predict_proba(X)
        return None
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        if self.task == "classification":
            y_proba = self.predict_proba(X_test)[:, 1]
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            metrics = {
                "mae": mean_absolute_error(y_test, y_pred),
                "mse": mse,
                "rmse": np.sqrt(mse),
                "r2": r2_score(y_test, y_pred)
            }
        
        return metrics
    
    def get_feature_importance(self, feature_names):
        if not self.trained:
            return None
        
        importances = self.model.feature_importances_
        importance_dict = {
            name: imp for name, imp in zip(feature_names, importances)
        }
        sorted_importance = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_importance
    
    def save(self, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "wb") as f:
            pickle.dump(self.model, f)
    
    def load(self, load_path):
        with open(load_path, "rb") as f:
            self.model = pickle.load(f)
        self.trained = True

