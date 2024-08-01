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
    def __init__(self, data_path, posterior_val=0.3, quality_val=0.1, posterior_length=500, peripheral_angle=90, detector_threshold=0.5):
        with open(os.path.join(data_path, 'annotations.json')) as f:
            self.data_dict = json.load(f)
        with open(os.path.join(data_path, 'fid_annotations.json')) as f:
            self.fid_annotations = json.load(f)
        self.fid_dict = self._gather_data()
        self.quality_val = quality_val
        self.posterior_length = posterior_length
        self.peripheral_angle = peripheral_angle
        self.posterior_val = posterior_val

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
                    if self._is_posterior(image_name):
                        quality_score *= self.posterior_val
                    else:
                        quality_score *= (1 - self.posterior_val)
                    quality_scores.append((quality_score, image_name))

                quality_scores.sort(reverse=True, key=lambda x: x[0])  # Sort based on quality score

                select_image = quality_scores[0][1]
                sequence_copy.remove(select_image)
                tar_sequence.append(select_image)
                if self._is_posterior(select_image):
                    poster.append(select_image)
                else:
                    periph.append(select_image)
            return tar_sequence
        else:
            raise ValueError(f"Unexpected sort method: {method}")

    # Calculate the detection acceleration rate
    def _get_accelerate(self, method='random'):
        success_detect_num_list = []
        success_detect_rate_list = []
        success_detect_cnt = 0
        fail_detect_cnt = 0
        
        cnt=10
        for fid in self.fid_annotations:
            success_detect = False
            if self.fid_annotations[fid] > 0:  # positive
                sorted_sequence = self._sort(sequence=self.fid_dict[fid], method=method)
                for i, image_name in enumerate(sorted_sequence, start=1):
                    data = self.data_dict[image_name]
                    if 'ROP_detect' not in data:
                        raise ValueError(f"{image_name} does not have ridge_seg")
                    # if self.detecor[image_name] and data['stage']>0:
                    if data['stage']>0:
                        if i==1:
                            cnt-=1
                            if cnt==0:
                                raise
                            print(f"{fid} only one recoginize with {str(len(sorted_sequence))}")
                            for image_name in sorted_sequence:
                                print(f"{image_name} stage: {self.data_dict[image_name]['stage']}")
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

def calculate_statistics(data):
    mean = np.mean(data)
    std = np.std(data)
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    return mean, std, median, q1, q3

results = []

posterior_vals = np.arange(0.0, 1.1, 0.1)[::-1]
quality_vals = np.arange(0.0, 1.1, 0.1)[::-1]
posterior_lengths = np.arange(1, 701, 100)[::-1]
peripheral_angles = np.arange(1, 141, 10)[::-1]


best_rate_median = {'value': None, 'params': None}
best_rate_q1 = {'value': None, 'params': None}
best_rate_q3 = {'value': None, 'params': None}
best_rate_mean = {'value': None, 'params': None}

# Initialize best metrics for base_number
best_number_median = {'value': None, 'params': None}
best_number_q1 = {'value': None, 'params': None}
best_number_q3 = {'value': None, 'params': None}
best_number_mean = {'value': None, 'params': None}

for posterior_val in posterior_vals:
    for quality_val in quality_vals:
        for posterior_length in posterior_lengths:
            for peripheral_angle in peripheral_angles:
                
                handler.posterior_val = posterior_val
                handler.quality_val = quality_val
                handler.posterior_length = posterior_length
                handler.peripheral_angle = peripheral_angle
                # handler.posterior_val = 0.0
                # handler.quality_val = 0.0
                # handler.posterior_length = posterior_length
                # handler.peripheral_angle = peripheral_angle
                base_number, base_rate = handler._get_accelerate(method='score')
                rate_mean, _, rate_median, rate_q1, rate_q3 = calculate_statistics(base_rate)
                number_mean, _, number_median, number_q1, number_q3 = calculate_statistics(base_number)
                
                results.append({
                    'posterior_val': posterior_val,
                    'quality_val': quality_val,
                    'posterior_length': posterior_length,
                    'peripheral_angle': peripheral_angle,
                    'rate_mean': rate_mean,
                    'rate_median': rate_median,
                    'rate_q1': rate_q1,
                    'rate_q3': rate_q3,
                    'number_mean': number_mean,
                    'number_median': number_median,
                    'number_q1': number_q1,
                    'number_q3': number_q3
                })
                
                if best_rate_median['value'] is None or rate_median < best_rate_median['value']:
                    best_rate_median['value'] = rate_median
                    best_rate_median['params'] = (posterior_val, quality_val, posterior_length, peripheral_angle)
                
                if best_rate_q1['value'] is None or rate_q1 < best_rate_q1['value']:
                    best_rate_q1['value'] = rate_q1
                    best_rate_q1['params'] = (posterior_val, quality_val, posterior_length, peripheral_angle)
                
                if best_rate_q3['value'] is None or rate_q3 < best_rate_q3['value']:
                    best_rate_q3['value'] = rate_q3
                    best_rate_q3['params'] = (posterior_val, quality_val, posterior_length, peripheral_angle)
                
                if best_rate_mean['value'] is None or rate_mean < best_rate_mean['value']:
                    best_rate_mean['value'] = rate_mean
                    best_rate_mean['params'] = (posterior_val, quality_val, posterior_length, peripheral_angle)

                # Evaluate and update best base_number metrics
                if best_number_median['value'] is None or number_median < best_number_median['value']:
                    best_number_median['value'] = number_median
                    best_number_median['params'] = (posterior_val, quality_val, posterior_length, peripheral_angle)
                
                if best_number_q1['value'] is None or number_q1 < best_number_q1['value']:
                    best_number_q1['value'] = number_q1
                    best_number_q1['params'] = (posterior_val, quality_val, posterior_length, peripheral_angle)
                
                if best_number_q3['value'] is None or number_q3 < best_number_q3['value']:
                    best_number_q3['value'] = number_q3
                    best_number_q3['params'] = (posterior_val, quality_val, posterior_length, peripheral_angle)
                
                if best_number_mean['value'] is None or number_mean < best_number_mean['value']:
                    best_number_mean['value'] = number_mean
                    best_number_mean['params'] = (posterior_val, quality_val, posterior_length, peripheral_angle)

print("对于base_rate:")
print(f"最优的中位数为: {best_rate_median['value']:.2f}，在参数组: {best_rate_median['params']}")
print(f"最优的Q1为: {best_rate_q1['value']:.2f}，在参数组: {best_rate_q1['params']}")
print(f"最优的Q3为: {best_rate_q3['value']:.2f}，在参数组: {best_rate_q3['params']}")
print(f"最优的平均数为: {best_rate_mean['value']:.2f}，在参数组: {best_rate_mean['params']}")

print("对于base_number:")
print(f"最优的中位数为: {best_number_median['value']:.2f}，在参数组: {best_number_median['params']}")
print(f"最优的Q1为: {best_number_q1['value']:.2f}，在参数组: {best_number_q1['params']}")
print(f"最优的Q3为: {best_number_q3['value']:.2f}，在参数组: {best_number_q3['params']}")
print(f"最优的平均数为: {best_number_mean['value']:.2f}，在参数组: {best_number_mean['params']}")
