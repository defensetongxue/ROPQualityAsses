import os
import json
import math
from configs import get_config

# 获取配置
args = get_config()

# 加载 annotations.json 文件
with open(os.path.join(args.data_path, 'annotations.json')) as f:
    data_dict = json.load(f)

# 图像的宽和高
image_width, image_height = 1600, 1200

# 中心点坐标
center_x, center_y = image_width / 2, image_height / 2

for image_name in data_dict:
    optic_position = data_dict[image_name]['optic_disc_pred']['position']
    x, y = optic_position[0], optic_position[1]
    
    # 转换为相对于中心点的坐标
    dx, dy = x - center_x, y - center_y
    
    # 计算角度（以x的正方向为0度，逆时针为正方向）
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360
    
    data_dict[image_name]['optic_disc_angle'] = angle

# 保存更新后的 annotations.json 文件
with open(os.path.join(args.data_path, 'annotations.json'), 'w') as f:
    json.dump(data_dict, f, indent=4)
