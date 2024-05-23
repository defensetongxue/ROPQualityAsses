import torch
from torch.utils.data import Dataset
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import Counter

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, dataset, smoothing=0.1):
        """
        Constructor for the LabelSmoothingCrossEntropy module.
        :param dataset: Dataset instance to calculate class weights.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.class_weights = self.calculate_class_weights(dataset)
        print()
    def calculate_class_weights(self, dataset):
        # Count the number of occurrences for each class
        class_counts = np.zeros(4, dtype=np.int)
        for _, label,_ in dataset:
            class_counts[label] += 1

        # Calculate weights for classes 1, 2, 3
        # Avoid division by zero
        class_counts[1:] = np.maximum(class_counts[1:], 1)
        weights = class_counts[1] / class_counts[1:]
        weights = np.array([1.0, 1.0] + list(weights))

        return torch.tensor(weights, dtype=torch.float32)
    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)

        # Apply class weights
        if self.class_weights is not None:
            weights = self.class_weights.to(x.device).view(-1, 1, 1)
            logprobs = logprobs * weights

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        return loss.mean()
    

class AdaptiveCrossEntropyLoss(nn.Module):
    def __init__(self, dataset, device='cpu', ignore_index=-100):
        """
        Constructor for AdaptiveCrossEntropyLoss.
        :param dataset: Dataset instance to calculate class weights.
        :param device: The device on which the computations are performed.
        :param ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
        """
        super(AdaptiveCrossEntropyLoss, self).__init__()
        self.device = device
        self.class_weights = self.calculate_class_weights(dataset).to(device)
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=ignore_index)
        print("Adaptive loss weight: ",self.class_weights)
        
    def calculate_class_weights(self, dataset):
        # Count the number of occurrences for each class
        label_counter = Counter()
        for _, label,_ in dataset:
            label_counter[label] += 1
        print(label_counter)
        # Calculate weights for classes 1, 2, and 3
        # Set weights for class 0 and 1 to 1.0
        class_weights = [1.0, 1.0]

        # Avoid division by zero and normalize weights for classes 2 and 3
        total_counts_1_2_3 = sum([label_counter[i] for i in range(1, 4)])
        for i in range(2, 4):
            weight = total_counts_1_2_3 / label_counter[i] if label_counter[i] > 0 else 1.0
            class_weights.append(weight)

        # Normalize weights such that the smallest weight is 1.0
        min_weight = min(class_weights)
        class_weights = [w / min_weight for w in class_weights]

        return torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, input, target):
        return self.cross_entropy_loss(input,target)