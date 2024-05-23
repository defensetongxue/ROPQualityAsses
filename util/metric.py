import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
import json
import os

class Metrics:
    def __init__(self, header="Main"):
        self.header = header
        self.score2label = {0: 0, 0.5: 1, 1: 2}
        self.reset()

    def reset(self):
        self.accuracy = 0
        self.auc = 0
        self.recall = []

    def update(self, predictions, probs, targets):
        # 先将概率值转换回标签
        
        self.accuracy = accuracy_score(targets, predictions)
        self.auc = roc_auc_score(targets, probs, multi_class='ovr')
        self.recall = []

        for i in range(max(targets) + 1):
            recall_value = recall_score(targets, predictions, labels=[i], average='macro')
            self.recall.append(round(recall_value, 4))

    def __str__(self):
        recall_str = ', '.join([f"Class {i}: {recall:.3f}" for i, recall in enumerate(self.recall)])
        return (f"[{self.header}] "
                f"Acc: {self.accuracy:.4f}, Auc: {self.auc:.4f}, Recall: {recall_str}")

    def _store(self, save_epoch, param, save_path='./record.json'):
        res = {
            "accuracy": round(self.accuracy, 4),
            "auc": round(self.auc, 4),
            "recall": self.recall,
            "save_epoch": save_epoch
        }

        # Check if the file exists and load its content if it does
        if os.path.exists(save_path):
            with open(save_path, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = []

        # Append the new data
        existing_data.append({
            "result":res,
            "param":param
        })

        # Save the updated data back to the file
        with open(save_path, 'w') as file:
            json.dump(existing_data, file, indent=4)

            