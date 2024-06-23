import os
import json
import numpy as np
from configs import get_config
import random  # Importing random for random sorting

# Get configuration
args = get_config()

class QualitySortHandler:
    def __init__(self, data_path, posterior_val=0.3,quality_val=0.1, posterior_length=500, peripheral_angle=90):
        with open(os.path.join(data_path, 'annotations.json')) as f:
            self.data_dict = json.load(f)
        with open(os.path.join(data_path, 'fid_annotations.json')) as f:
            self.fid_annotations = json.load(f)
        self.fid_dict = self._gather_data()
        self.quality_val = quality_val
        self.posterior_length = posterior_length
        self.peripheral_angle = peripheral_angle
        self.posterior_val= posterior_val

        self.use_posterior=None
        self.use_rever=False
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
        print(angle1,angle2)
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
        if self.use_rever:
            if self.use_posterior:
                pos_val=0
            else:
                pos_val=1
        else:
            pos_val=self.posterior_val
        if self._is_posterior(image_name):
            return self._posterior_score(image_name, posterior_list)*pos_val
        else:
            return self._peripheral_score(image_name, peripheral_list)*(1-pos_val)

    # Sort the sequence based on the specified method
    def _sort(self, sequence, method='random'):
        if method == 'random':
            random.shuffle(sequence)
            return sequence
        elif method == 'score':
            tar_sequence = []
            poster = []
            periph = []
            while sequence:
                quality_scores = []
                for image_name in sequence:
                    quality_score = self.data_dict[image_name]['qualityScorePred'] * self.quality_val + \
                                    self._cal_angle_score(image_name, posterior_list=poster, peripheral_list=periph) * (1 - self.quality_val)
                    quality_scores.append((quality_score, image_name))
                
                quality_scores.sort(reverse=True, key=lambda x: x[0])  # Sort based on quality score

                select_image = quality_scores[0][1]
                sequence.remove(select_image)
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
        for fid in self.fid_annotations:
            if self.fid_annotations[fid] > 0:  # positive
                sorted_sequence = self._sort(sequence=self.fid_dict[fid], method=method)
                for i, image_name in enumerate(sorted_sequence, start=1):
                    data = self.data_dict[image_name]
                    if 'ROP_detect' not in data:
                        raise ValueError(f"{image_name} does not have ridge_seg")
                    if data['ROP_detect'] > 0.5:
                        success_detect_num_list.append(i)
                        success_detect_rate_list.append(i / len(sorted_sequence))
                        break  # Stop after first successful detection
        return np.mean(success_detect_num_list), np.mean(success_detect_rate_list),success_detect_num_list,success_detect_rate_list

data_path='../quality_annotation'

hander= QualitySortHandler(data_path)
res_mean,res_mean_r,l1,l2=hander._get_accelerate(method='score')
print(max(l1),max(l2))

print(res_mean,res_mean_r)