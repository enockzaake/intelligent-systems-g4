import json
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class ModelEvaluator:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def add_result(self, model_name, task, metrics):
        if model_name not in self.results:
            self.results[model_name] = {}
        
        self.results[model_name][task] = metrics
    
    def save_results(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def create_comparison_report(self):
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        for model_name, tasks in self.results.items():
            report_lines.append(f"\n{'=' * 80}")
            report_lines.append(f"Model: {model_name}")
            report_lines.append(f"{'=' * 80}")
            
            for task, metrics in tasks.items():
                report_lines.append(f"\nTask: {task.upper()}")
                report_lines.append("-" * 40)
                
                for metric_name, value in metrics.items():
                    if metric_name != "confusion_matrix":
                        if isinstance(value, float):
                            report_lines.append(f"  {metric_name}: {value:.4f}")
                        else:
                            report_lines.append(f"  {metric_name}: {value}")
                
                if "confusion_matrix" in metrics:
                    report_lines.append(f"\n  Confusion Matrix:")
                    cm = metrics["confusion_matrix"]
                    for row in cm:
                        report_lines.append(f"    {row}")
        
        report_lines.append("\n" + "=" * 80)
        
        report_text = "\n".join(report_lines)
        
        report_path = self.results_dir / "evaluation_report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nReport saved to {report_path}")
        
        return report_text
    
    def create_comparison_table(self):
        classification_data = []
        regression_data = []
        
        for model_name, tasks in self.results.items():
            if "classification" in tasks:
                metrics = tasks["classification"]
                classification_data.append({
                    "Model": model_name,
                    "Accuracy": f"{metrics.get('accuracy', 0):.4f}",
                    "Precision": f"{metrics.get('precision', 0):.4f}",
                    "Recall": f"{metrics.get('recall', 0):.4f}",
                    "F1-Score": f"{metrics.get('f1_score', 0):.4f}",
                    "ROC-AUC": f"{metrics.get('roc_auc', 0):.4f}"
                })
            
            if "regression" in tasks:
                metrics = tasks["regression"]
                regression_data.append({
                    "Model": model_name,
                    "MAE": f"{metrics.get('mae', 0):.4f}",
                    "MSE": f"{metrics.get('mse', 0):.4f}",
                    "RMSE": f"{metrics.get('rmse', 0):.4f}",
                    "R²": f"{metrics.get('r2', 0):.4f}"
                })
        
        comparison = {}
        
        if classification_data:
            clf_df = pd.DataFrame(classification_data)
            comparison["classification"] = clf_df
            
            clf_path = self.results_dir / "classification_comparison.csv"
            clf_df.to_csv(clf_path, index=False)
            print(f"\nClassification Comparison:")
            print(clf_df.to_string(index=False))
            print(f"Saved to {clf_path}")
        
        if regression_data:
            reg_df = pd.DataFrame(regression_data)
            comparison["regression"] = reg_df
            
            reg_path = self.results_dir / "regression_comparison.csv"
            reg_df.to_csv(reg_path, index=False)
            print(f"\nRegression Comparison:")
            print(reg_df.to_string(index=False))
            print(f"Saved to {reg_path}")
        
        return comparison
    
    def plot_model_comparison(self):
        classification_metrics = {}
        regression_metrics = {}
        
        for model_name, tasks in self.results.items():
            if "classification" in tasks:
                metrics = tasks["classification"]
                for metric_name, value in metrics.items():
                    if metric_name != "confusion_matrix":
                        if metric_name not in classification_metrics:
                            classification_metrics[metric_name] = {}
                        classification_metrics[metric_name][model_name] = value
            
            if "regression" in tasks:
                metrics = tasks["regression"]
                for metric_name, value in metrics.items():
                    if metric_name not in regression_metrics:
                        regression_metrics[metric_name] = {}
                    regression_metrics[metric_name][model_name] = value
        
        if classification_metrics:
            metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
            available_metrics = [m for m in metrics_to_plot if m in classification_metrics]
            
            if available_metrics:
                fig, axes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 4))
                
                if len(available_metrics) == 1:
                    axes = [axes]
                
                for idx, metric in enumerate(available_metrics):
                    models = list(classification_metrics[metric].keys())
                    values = list(classification_metrics[metric].values())
                    
                    axes[idx].bar(models, values, color='steelblue')
                    axes[idx].set_title(metric.replace('_', ' ').title())
                    axes[idx].set_ylabel('Score')
                    axes[idx].set_ylim([0, 1])
                    axes[idx].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plot_path = self.results_dir / "classification_comparison.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Classification comparison plot saved to {plot_path}")
                plt.close()
        
        if regression_metrics:
            metrics_to_plot = ["mae", "rmse", "r2"]
            available_metrics = [m for m in metrics_to_plot if m in regression_metrics]
            
            if available_metrics:
                fig, axes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 4))
                
                if len(available_metrics) == 1:
                    axes = [axes]
                
                for idx, metric in enumerate(available_metrics):
                    models = list(regression_metrics[metric].keys())
                    values = list(regression_metrics[metric].values())
                    
                    color = 'seagreen' if metric == 'r2' else 'tomato'
                    axes[idx].bar(models, values, color=color)
                    axes[idx].set_title(metric.upper())
                    axes[idx].set_ylabel('Value')
                    axes[idx].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plot_path = self.results_dir / "regression_comparison.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Regression comparison plot saved to {plot_path}")
                plt.close()
    
    def plot_feature_importance(self, model, feature_names, model_name, top_n=20):
        importance = model.get_feature_importance(feature_names)
        
        if importance is None:
            print(f"Feature importance not available for {model_name}")
            return
        
        top_features = importance[:top_n]
        features = [item[0] for item in top_features]
        importances = [item[1] for item in top_features]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importances, color='steelblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plot_path = self.results_dir / f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {plot_path}")
        plt.close()
    
    def load_results(self, filepath):
        with open(filepath, "r") as f:
            self.results = json.load(f)

