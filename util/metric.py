import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
import json
import os

from sklearn.metrics import accuracy_score, recall_score, mean_squared_error, mean_absolute_error, r2_score

class Metrics:
    def __init__(self, header="Main"):
        self.header = header
        self.score2label = {0: 0, 0.5: 1, 1: 2}
        self.reset()

    def reset(self):
        self.accuracy = 0
        self.recall = []
        self.mse = 0
        self.mae = 0
        self.r2 = 0
    def get_pred_label(self,predictions_probs):
        predictions=[]
        for prob in predictions_probs:
            if prob < 0.67:
                predictions.append(0)
            elif prob < 1.33:
                predictions.append(1)
            else:
                predictions.append(2)
        return predictions
    def get_onehot_encoder(self,predictions):
        return np.eye(2)[predictions]
    def update(self, predictions, targets):
        # 将回归值四舍五入到最近的标签
        rounded_predictions = self.get_pred_label(predictions)
        # rounded_predictions = np.clip(rounded_predictions, 0, 2)

        self.accuracy = accuracy_score(targets, rounded_predictions)
        self.mse = mean_squared_error(targets, predictions)
        self.mae = mean_absolute_error(targets, predictions)
        self.r2 = r2_score(targets, predictions)
        self.recall = []

        for i in range(int(max(targets) + 1)):
            recall_value = recall_score(targets, rounded_predictions, labels=[i], average='macro')
            self.recall.append(round(recall_value, 4))

    def __str__(self):
        recall_str = ', '.join([f"Class {i}: {recall:.3f}" for i, recall in enumerate(self.recall)])
        return (f"[{self.header}] "
                f"Acc: {self.accuracy:.4f}, MSE: {self.mse:.4f}, "
                f"MAE: {self.mae:.4f}, R2: {self.r2:.4f}, Recall: {recall_str}")


    def _store(self, save_epoch, param, save_path='./record.json'):
        def handle_nan(value):
            return 0.0 if np.isnan(value) else value
        res = {
            "accuracy": float(round(handle_nan(self.accuracy), 4)),
            "mse": float(round(handle_nan(self.mse), 4)),
            "mae": float(round(handle_nan(self.mae), 4)),
            "r2": float(round(handle_nan(self.r2), 4)),
            "recall": [float(handle_nan(recall)) for recall in self.recall],
            "save_epoch": save_epoch
        }
        print(res)
        # Check if the file exists and load its content if it does
        if os.path.exists(save_path):
            with open(save_path, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = []

        # Append the new data
        existing_data.append({
            "result": res,
            "param": param
        })

        # Save the updated data back to the file
        with open(save_path, 'w') as file:
            json.dump(existing_data, file, indent=4)
            