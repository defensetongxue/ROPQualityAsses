import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive environments

data_path = '../quality_annotation'

# Load data
with open(os.path.join(data_path, 'annotations.json')) as f:
    data_dict = json.load(f)
with open(os.path.join(data_path, 'fid_annotations.json')) as f:
    fid_annotations = json.load(f)
with open('detector/efficientnet_b7.json') as f:
    detector=json.load(f)
# Prepare data
follow_up_split = {}
for image_name in data_dict:
    fid = data_dict[image_name]['fid']
    if fid not in follow_up_split:
        follow_up_split[fid] = []
    follow_up_split[fid].append(image_name)
for fid in follow_up_split:
    follow_up_split[fid] = sorted(follow_up_split[fid], key=lambda x: int(x[:-4]))

x = []
y = []
for fid in fid_annotations:
    if fid_annotations[fid] > 0:
        for image_name in follow_up_split[fid]:
            if  data_dict[image_name]['stage'] >0:
                x.append(data_dict[image_name]['qualityLevel'])
                # y.append(0 if data_dict[image_name]['stage'] == 0 else 1)
                y.append(detector[image_name])

# Calculate correlation coefficient
correlation_coefficient = np.corrcoef(x, y)[0, 1]

# Print correlation coefficient
print(f"Correlation Coefficient: {correlation_coefficient:.2f}")

# Interpretation of correlation coefficient
if correlation_coefficient > 0:
    interpretation = "正相关: 当一个变量增加时，另一个变量也增加。"
elif correlation_coefficient < 0:
    interpretation = "负相关: 当一个变量增加时，另一个变量减少。"
else:
    interpretation = "不相关: 两个变量之间没有明显的线性关系。"

print(interpretation)
