import os
import json
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from math import sin, cos, radians
from util.tools import visual_optic_disc
import matplotlib
matplotlib.use('Agg')  # 使用无头后端
import matplotlib.pyplot as plt
class Juedge():
    def __init__(self, mask_path='./mask.png', threshold=40, check_number=48):
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        mask[mask > 0] = 1
        self.mask = mask
        self.threshold = threshold
        self.check_list = []
        # 建立 check_number 个检测点，检测四周有没有位于边缘的点
        angle_step = 360 / check_number
        for i in range(check_number):
            angle = radians(i * angle_step)
            self.check_list.append((int(threshold * sin(angle)), int(threshold * cos(angle))))
        self.check_number = check_number
        self.h, self.w = self.mask.shape

    def is_complete(self, position):
        x, y = position
        if x < 0 or y < 0 or x >= self.w or y >= self.h or self.mask[y, x] < 1:
            return False
        for x_plus, y_plus in self.check_list:
            x_check, y_check = x + x_plus, y + y_plus
            if x_check < 0 or y_check < 0 or x_check >= self.w or y_check >= self.h or self.mask[y_check, x_check] < 1:
                return False
        return True

    def set_threshold(self, threshold):
        self.threshold = threshold
        self.check_list = []
        angle_step = 360 / self.check_number
        for i in range(self.check_number):
            angle = radians(i * angle_step)
            self.check_list.append((int(threshold * sin(angle)), int(threshold * cos(angle))))

class PredAngleType():
    def __init__(self, data_path, threshold):     
        self.data_path = data_path
        with open(os.path.join(data_path, 'annotations.json')) as f:
            self.data_dict = json.load(f)
        self.judger = Juedge(threshold=threshold)
        self.visual_dir = './experiments/error_type'
        os.makedirs(self.visual_dir, exist_ok=True)
        for folder in ['0', '1']:
            os.makedirs(os.path.join(self.visual_dir, folder), exist_ok=True)

    def test_threshold(self, threshold, load_pred=False):
        self.judger.set_threshold(threshold)
        labels = []
        preds = []
        for image_name in self.data_dict:
            data = self.data_dict[image_name]
            if 'optic_disc_pred' not in data:
                raise ValueError(f"{image_name} does not have optic disc prediction")

            optic_disc = data['optic_disc_pred']
            if optic_disc['distance'] != 'visible':
                angleType = 0
            else:
                if self.judger.is_complete(optic_disc['position']):
                    angleType = 1
                else:
                    angleType = 0
                    
            labels.append(data['angleType'])
            preds.append(angleType)
            # if angleType != data['angleType']:
            #     visual_optic_disc('/mnt/d/dataset_ROP/images/' + image_name, optic_disc['position'],
            #                       os.path.join(self.visual_dir, str(data["angleType"]), image_name))
            if load_pred:
                self.data_dict[image_name]['angleType_pred'] = angleType

        if load_pred:
            with open(os.path.join(self.data_path, 'annotations.json'), 'w') as f:
                json.dump(self.data_dict, f)
                
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else float('nan')
        recall_0 = recall_score(labels, preds, pos_label=0)
        recall_1 = recall_score(labels, preds, pos_label=1)
        
        return acc, auc, recall_0, recall_1


data_path = '../quality_annotation'
threshold = 20
load_annote = False

preder = PredAngleType(data_path, threshold)

thresholds = np.arange(5, 100, 5)
accuracies = []
aucs = []
recalls_0 = []
recalls_1 = []

for threshold in thresholds:
    acc, auc, recall_0, recall_1 = preder.test_threshold(threshold)
    accuracies.append(acc)
    aucs.append(auc)
    recalls_0.append(recall_0)
    recalls_1.append(recall_1)
    print(f"{threshold}: acc:{acc:.4f} auc:{auc:.4f} recall_0:{recall_0:.4f} recall_1:{recall_1:.4f}")

# 绘制曲线图并保存
# 确保保存路径存在
save_dir = './experiments/angleType'
os.makedirs(save_dir, exist_ok=True)
# 颜色设置
color = (49/255, 102/255, 88/255)

def plot_bars_with_max_label(thresholds, values, ylabel, title, filename):
    plt.figure()
    min_val, max_val = min(values), max(values)
    # 只显示超过0.6的部分
    plot_values = [(max(0, (x - 0.6) / 0.4)) if x >= 0.6 else 0 for x in values]
    alphas = [0.6 + 0.4 * (x - min_val) / (max_val - min_val) for x in values]
    bars = plt.bar(thresholds, plot_values, width=5, color=[(*color, alpha) for alpha in alphas])
    
    # 只在最高的柱子上标注分值
    max_idx = np.argmax(values)
    max_bar = bars[max_idx]
    height = values[max_idx]
    plt.text(max_bar.get_x() + max_bar.get_width() / 2, plot_values[max_idx], f'{height:.3f}', ha='center', va='bottom')
    
    plt.xlabel('Threshold')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1)
    plt.yticks([0.0, 0.5, 1.0], ['0.6', '0.8', '1.0'])
    # 去掉上方和右边的边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

# 绘制 Accuracy 柱状图
plot_bars_with_max_label(thresholds, accuracies, 'Accuracy', 'Accuracy at Different Thresholds', 'accuracy.png')

# 绘制 AUC 柱状图
plot_bars_with_max_label(thresholds, aucs, 'AUC', 'AUC at Different Thresholds', 'auc.png')

# # 绘制 Recall for class 0 和 Recall for class 1 的图
# plt.figure()
# plt.plot(thresholds, recalls_0, label='Recall for class 0', color='r', marker='o')
# plt.plot(thresholds, recalls_1, label='Recall for class 1', color='m', marker='o')

# 适当降低点的高度
plot_recalls_0 = [(max(0, (x - 0.6) / 0.4)) if x >= 0.6 else 0 for x in recalls_0]
plot_recalls_1 = [(max(0, (x - 0.6) / 0.4)) if x >= 0.6 else 0 for x in recalls_1]
plt.plot(thresholds, plot_recalls_0, label='Recall for class 0', color='r', marker='o')
plt.plot(thresholds, plot_recalls_1, label='Recall for class 1', color='m', marker='o')

plt.xlabel('Threshold')
plt.ylabel('Recall')
plt.title('Recall for class 0 and class 1 at Different Thresholds')
plt.legend()
plt.ylim(0, 1)
plt.yticks([0.0, 0.5, 1.0], ['0.6', '0.8', '1.0'])
# 去掉上方和右边的边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig(os.path.join(save_dir, 'recall.png'), dpi=300)
plt.close()