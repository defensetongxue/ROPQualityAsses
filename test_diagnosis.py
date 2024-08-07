import os
import json
import numpy as np
import random  # Importing random for random sorting
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive environments
import matplotlib.pyplot as plt
from configs import get_config

# Get configuration
args = get_config()
class QualityScoreNormer:
    def __init__(self, data_dict):
        score_list = []
        for image_name in data_dict:
            score_list.append(data_dict[image_name]['qualityScorePred'])
        self.max = max(score_list)
        self.min = min(score_list)
    
    def norm(self, val):
        return (val - self.min) / (self.max - self.min)

class QualitySortHandler:
    def __init__(self, data_path, peripheral_val=0.1, quality_val=0.1, posterior_length=500, peripheral_angle=90, detector_threshold=0.5):
        with open(os.path.join(data_path, 'annotations.json')) as f:
            self.data_dict = json.load(f)
        with open(os.path.join(data_path, 'fid_annotations.json')) as f:
            self.fid_annotations = json.load(f)
        self.fid_dict = self._gather_data()
        self.quality_val = quality_val
        self.posterior_length = posterior_length
        self.peripheral_angle = peripheral_angle
        self.peripheral_val = peripheral_val

        self.use_posterior = None
        self.use_rever = False
        
        self._load_detector(detector_threshold)
        self.normer = QualityScoreNormer(self.data_dict)

    # Gather data and split by fid
    def _gather_data(self):
        follow_up_split = {}
        for image_name in self.data_dict:
            fid = self.data_dict[image_name]['fid']
            if fid not in follow_up_split:
                follow_up_split[fid] = []
            follow_up_split[fid].append(image_name)
        for fid in follow_up_split:
            follow_up_split[fid] = sorted(follow_up_split[fid], key=lambda x: int(x[:-4]))
        return follow_up_split

    # Check if the image is posterior
    def _is_posterior(self, image_name):
        return self.data_dict[image_name]['optic_disc_pred']['distance'] == 'visible'

    # Get the optic disc position of the image
    def _get_optic_disc_position(self, image_name):
        return self.data_dict[image_name]['optic_disc_pred']['position']

    # Get the optic disc angle of the image
    def _get_optic_disc_angle(self, image_name):
        return self.data_dict[image_name]['optic_disc_angle']

    # Calculate the posterior score
    def _posterior_score(self, image_name, selected_list):
        res_score = 1
        for select_image in selected_list:
            score = np.linalg.norm(np.array(self._get_optic_disc_position(image_name)) - np.array(self._get_optic_disc_position(select_image))) / self.posterior_length
            res_score = min(res_score, score)
        return res_score

    # Calculate the relative angle between two angles
    def _cal_related_angle(self, angle1, angle2):
        if angle1 < angle2:
            a, b = angle1, angle2
        else:
            a, b = angle2, angle1
        return min(a + 360 - b, b - a)

    # Calculate the peripheral score
    def _peripheral_score(self, image_name, selected_list):
        res_score = 1
        for select_image in selected_list:
            score = self._cal_related_angle(self._get_optic_disc_angle(image_name), self._get_optic_disc_angle(select_image)) / self.peripheral_angle
            res_score = min(res_score, score)
        return res_score

    # Calculate the angle score based on whether the image is posterior or peripheral
    def _cal_angle_score(self, image_name, posterior_list, peripheral_list):
        if self._is_posterior(image_name):
            return self._posterior_score(image_name, posterior_list)
        else:
            return self._peripheral_score(image_name, peripheral_list)

    # Load detector if the predict possibility is greater than the threshold
    def _load_detector(self, threshold=0.5):
        self.detecor = {}
        for image_name in self.data_dict:
            if self.data_dict[image_name]['ROP_detect'] > threshold:
                self.detecor[image_name] = True
            else:
                self.detecor[image_name] = False

    # Sort the sequence based on the specified method
    def _sort(self, sequence, method='random'):
        sequence_copy = sequence.copy()  # 使用副本进行排序操作
        if method == 'random':
            random.shuffle(sequence_copy)
            return sequence_copy
        elif method == 'score':
            tar_sequence = []
            poster = []
            periph = []
            random.shuffle(sequence_copy) #  防止保序排序
            while sequence_copy:
                quality_scores = []
                for image_name in sequence_copy:
                    quality_score = self.normer.norm(self.data_dict[image_name]['qualityScorePred']) * self.quality_val + \
                                    self._cal_angle_score(image_name, posterior_list=poster, peripheral_list=periph) * (1 - self.quality_val)
                    
                    
                    if not self._is_posterior(image_name):
                        quality_score += self.peripheral_val
                    quality_scores.append((quality_score, image_name))
                quality_scores.sort(reverse=True, key=lambda x: x[0])  # Sort based on quality score
                # print(quality_scores)
                select_image = quality_scores[0][1]
                sequence_copy.remove(select_image)
                tar_sequence.append(select_image)
                # raise
                if self._is_posterior(select_image):
                    poster.append(select_image)
                else:
                    periph.append(select_image)
            return tar_sequence
        else:
            raise ValueError(f"Unexpected sort method: {method}")
    
    def calculate_statistics(self,data):
        mean = np.mean(data)
        std = np.std(data)
        median = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        r1 = len([i for i in data if i <= 1]) / len(data)
        r2 = len([i for i in data if i <= 2]) / len(data)
        r3 = len([i for i in data if i <= 3]) / len(data)
        return mean, std, median, q1, q3, r1, r2, r3

    # Calculate the detection acceleration rate
    def _get_accelerate(self, method='random'):
        success_detect_num_list = []
        success_detect_rate_list = []
        success_detect_cnt = 0
        fail_detect_cnt = 0
        
        for fid in self.fid_annotations:
            success_detect = False
            if self.fid_annotations[fid] > 0:  # positive
                sorted_sequence = self._sort(sequence=self.fid_dict[fid], method=method)
                for i, image_name in enumerate(sorted_sequence, start=1):
                    data = self.data_dict[image_name]
                    if 'ROP_detect' not in data:
                        raise ValueError(f"{image_name} does not have ridge_seg")
                    if data['stage'] > 0:
                        success_detect_num_list.append(i)
                        success_detect_rate_list.append(i / len(sorted_sequence))
                        success_detect = True
                        break  # Stop after first successful detection
                if success_detect:
                    success_detect_cnt += 1
                else:
                    fail_detect_cnt += 1
        return success_detect_num_list, success_detect_rate_list

data_path = '../quality_annotation'

handler = QualitySortHandler(data_path)

base_number_list = []
base_rate_list = []
for _ in range(10):
    base_number, base_rate = handler._get_accelerate(method='random')
    base_number_list.extend(base_number)
    base_rate_list.extend(base_rate)

base_number_stats = handler.calculate_statistics(base_number_list)
base_rate_stats = handler.calculate_statistics(base_rate_list)

# TODO 2 增加r1,r2,r3的打印
print(f"Base Number - Mean: {base_number_stats[0]:.2f}, Std: {base_number_stats[1]:.2f}, Median: {base_number_stats[2]:.2f}, Q1: {base_number_stats[3]:.2f}, Q3: {base_number_stats[4]:.2f}")
print(f"Base Rate - Mean: {base_rate_stats[0]:.2f}, Std: {base_rate_stats[1]:.2f}, Median: {base_rate_stats[2]:.2f}, Q1: {base_rate_stats[3]:.2f}, Q3: {base_rate_stats[4]:.2f}")
print(f"Base Number - r1: {base_number_stats[5]:.2f}, r2: {base_number_stats[6]:.2f}, r3: {base_number_stats[7]:.2f}")

results = []
for value in np.arange(0.0, 1.1, 0.1):
    handler.quality_val = value
    
    score_number = []
    score_rate = []
    for _ in range(10):
        score_number_, score_rate_ = handler._get_accelerate(method='score')
        score_number.extend(score_number_)
        score_rate.extend(score_rate_)
    results.append((value, score_number, score_rate))

# 绘制箱型图
fig, ax = plt.subplots(figsize=(10, 6))

# 将宽度分为13份
positions = [(i + 1) / 13 for i in range(len(results))]
for i, result in enumerate(results):
    value, _, score_rate = result
    ax.boxplot(score_rate, positions=[positions[i]], widths=0.05, showfliers=False)

# 绘制 base rate 的箱型图
ax.boxplot(base_rate_list, positions=[12 / 13], widths=0.05, showfliers=False)

ax.set_xlabel('Quality Value')
ax.set_ylabel('Detection Rate')
ax.set_title('Detection Rate vs. Quality Value')

# 设置 x 轴刻度
ax.set_xticks(positions + [12 / 13])
xticklabels = [f'{x:.1f}' for x in np.arange(0.0, 1.1, 0.1)] + ['Random']
ax.set_xticklabels(xticklabels)
# 设置 x 轴限制
ax.set_xlim(0, 1)

# 创建保存目录
save_dir = './experiments/detect_result'
os.makedirs(save_dir, exist_ok=True)

plt.savefig(os.path.join(save_dir, 'detection_rate_boxplot.png'), dpi=300)
plt.close()

r1_values = []
r2_values = []
r3_values = []

#TODO 2 找到 r1,r2,r3最高的时候对应的value，在后面打印
for value, score_number, _ in results:
    stats = handler.calculate_statistics(score_number)
    r1_values.append(stats[5])
    r2_values.append(stats[6])
    r3_values.append(stats[7])

# 添加 base 的 r1, r2, r3 值
r1_values.append(base_number_stats[5])
r2_values.append(base_number_stats[6])
r3_values.append(base_number_stats[7])

# 绘制 r1, r2, r3 变化的折线图
fig, ax = plt.subplots(figsize=(10, 6))
x_values = [f'{x:.1f}' for x in np.arange(0.0, 1.1, 0.1)] + ['Random']
ax.plot(x_values, r1_values, label='r1')
ax.plot(x_values, r2_values, label='r2')
ax.plot(x_values, r3_values, label='r3')
ax.set_xlabel('Quality Value')
ax.set_ylabel('Rate')
ax.set_title('Rate vs. Quality Value')
ax.legend()

# 保存图像
plt.savefig(os.path.join(save_dir, 'r1_r2_r3.png'), dpi=300)
plt.close()

# 找到 r1, r2, r3 最高值对应的 value
max_r1_value = x_values[r1_values.index(max(r1_values))]
max_r2_value = x_values[r2_values.index(max(r2_values))]
max_r3_value = x_values[r3_values.index(max(r3_values))]

print(f"Max r1 value: {max_r1_value}")
print(f"Max r2 value: {max_r2_value}")
print(f"Max r3 value: {max_r3_value}")

print(f"Results saved in {save_dir}")
