import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import Counter
import json,os
def calculate_recall(labels, preds, class_id=None):
    """
    Calculate recall for a specified class or for the positive label in a multi-class task.
    
    Args:
    labels (np.array): Array of true labels.
    preds (np.array): Array of predicted labels.
    class_id (int or None): Class ID for which to calculate recall. If None, calculate recall for the positive label.
    
    Returns:
    float: Recall for the specified class or for the positive label.
    """
    if class_id is not None:
        true_class = labels == class_id
        predicted_class = preds == class_id
    else:
        true_class = labels > 0
        predicted_class = preds > 0

    true_positives = np.sum(true_class & predicted_class)
    false_negatives = np.sum(true_class & ~predicted_class)

    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall

class Metrics:
    def __init__(self, dataset,header="Main" ):
        self.class_weights = self.calculate_class_weights(dataset)
        print(self.class_weights)
        self.reset()
        self.header=header
    def reset(self):
        self.accuracy = 0
        self.auc = 0
        self.recall_1 = 0
        self.recall_2 = 0
        self.recall_3 = 0
        self.recall_pos=0
        self.average_recall = 0
    def calculate_class_weights(self, dataset):
        # Calculate the distribution of classes 1, 2, and 3 in the dataset
        class_counter = Counter()
        for _, label,_ in dataset:
            if label in [1, 2, 3]:
                class_counter[label] += 1

        # Calculate the weights for each class
        total_count = sum(class_counter.values())
        class_weights = {i: (class_counter[i]/total_count ) if class_counter[i] > 0 else 0 for i in [1, 2, 3]}

        # Normalize weights
        # min_weight = min(filter(lambda x: x > 0, class_weights.values()), default=1)
        # class_weights = {i: weight / min_weight for i, weight in class_weights.items()}

        return class_weights

    def update(self, predictions, probs, targets):
        self.accuracy = accuracy_score(targets, predictions)
        # self.auc= roc_auc_score(targets,predictions)
        self.auc = roc_auc_score(targets, probs, multi_class='ovr')
        self.recall_1 = calculate_recall(targets, predictions, class_id=1)
        self.recall_2 = calculate_recall(targets, predictions, class_id=2)
        self.recall_3 = calculate_recall(targets, predictions, class_id=3)
        self.recall_pos=calculate_recall(targets,predictions)
        # # Compute weighted average recall
        # self.average_recall = sum(self.class_weights[i] * recall for i, recall in zip([1, 2, 3], [self.recall_1, self.recall_2, self.recall_3]))

    def __str__(self):
        return (f"[{self.header}] "
                f"Acc: {self.accuracy:.4f}, Auc: {self.auc:.4f}, "
                f"Recall1: {self.recall_1:.4f}, Recall2: {self.recall_2:.4f}, "
                f"Recall3: {self.recall_3:.4f}, RecallAvg: {self.average_recall:.4f}, RecallPos: {self.recall_pos:.4f} ")
    
    def _store(self,key, split_name,save_epoch,param, save_path='./record.json'):
        res  =  {
                "accuracy": round(self.accuracy, 4),
                "auc": round(self.auc, 4),
                "recall_1": round(self.recall_1, 4),
                "recall_2": round(self.recall_2, 4),
                "recall_3": round(self.recall_3, 4),
                "recall_pos": round(self.recall_pos, 4),
                "average_recall": round(self.average_recall, 4),
                "save_epoch":save_epoch
            }
        

        # Check if the file exists and load its content if it does
        if os.path.exists(save_path):
            with open(save_path, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = {}

        # Append the new data
        if key not in existing_data:
            existing_data[key]={
                "param":param,
                "result":{}
            }
        existing_data[key]["result"][split_name]=res

        # Save the updated data back to the file
        with open(save_path, 'w') as file:
            json.dump(existing_data, file, indent=4)
            
            