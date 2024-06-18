import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from util.tools import get_color
from configs import get_config
args=get_config()
# Set paths
save_dir = './experiments/record_figure/distribute'
font_path = './arial.ttf'
font_size = 15  # 增大字体大小
color_list = get_color("xinhai", 5)

# Ensure color values are normalized to [0, 1] for matplotlib
color_list = np.array(color_list) / 255.0

# Create directory to save results
os.makedirs(save_dir, exist_ok=True)

# Load local font
prop = FontProperties(fname=font_path, size=14)
y_tick_prop = FontProperties(fname=font_path, size=18)

# Load data from JSON file
with open(('../autodl-tmp/dataset_ROP/annotations.json')) as f:
    data_dict = json.load(f)

# Count the data
angle_type = args.angle_type
data_cnt = [0, 0, 0]
for image_name in data_dict:
    data = data_dict[image_name]
    if data['angleType'] == angle_type:
        data_cnt[data['qualityLevel']] += 1

# Inner pie chart data (combined class 1 and 2)
inner_sizes = [data_cnt[0], data_cnt[1] + data_cnt[2]]
inner_colors = color_list[-2:]

# Outer pie chart data (three classes)
outer_sizes = data_cnt
outer_colors = color_list[:3]

fig, ax = plt.subplots(figsize=(8, 8))

# Draw inner pie chart with smaller radius
ax.pie(
    inner_sizes, colors=inner_colors, startangle=90,
    radius=0.5
)

# Draw outer pie chart with larger radius and adjust wedge width
ax.pie(
    outer_sizes, colors=outer_colors, startangle=90,
    radius=0.7, wedgeprops=dict(width=0.3, edgecolor='w')
)

# Save the figure
plt.savefig(os.path.join(save_dir, f'{str(angle_type)}.png'), bbox_inches='tight', dpi=300)

# Close the figure to free up memory
plt.close()
print(data_cnt)
all_number=data_cnt[0]+data_cnt[1]+data_cnt[2]
print(data_cnt[0]/all_number,data_cnt[1]/all_number,data_cnt[2]/all_number)
print((data_cnt[1]+data_cnt[2])/all_number)

if os.path.exists(os.path.join(save_dir, 'record.json')):
    with open(os.path.join(save_dir, 'record.json')) as f:
        record=json.load(f)
else:
    record={}
record[str(angle_type)]={
    "data_number":all_number,
    "0_1_2_number":data_cnt
}

with open(os.path.join(save_dir, 'record.json'),'w') as f:
    json.dump(record,f,indent=4)