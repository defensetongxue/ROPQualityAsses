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
save_dir = './experiments/record_figure/data_condition'
font_path = './arial.ttf'
font_size = 15  # 增大字体大小

prop = FontProperties(fname=font_path, size=14)
color_list = get_color("xinhai", 5)

# Ensure color values are normalized to [0, 1] for matplotlib
color_list = np.array(color_list) / 255.0

# Create directory to save results
os.makedirs(save_dir, exist_ok=True)

# Load local font
y_tick_prop = FontProperties(fname=font_path, size=18)

res={
    "follow_up_number":0,
    "eye_number":453,
    "both_eye":0,
    "single_eye":0,
    "eye_min_number":0,
    "eye_max_number":0,
    "eye_average_number":0,
    "positive_eye":0,
    "positve_follow_up":0,
    "angle_type":{
        "0":{
            "name":"peripheral",
            "data_number":0,
            "positive_number":0,
            "quality_level":{
                "0":{
                    "data_number":0,
                    "positive_number":0
                },
                "1":{
                    "data_number":0,
                    "positive_number":0
                },
                "2":{
                    "data_number":0,
                    "positive_number":0
                }
            }
        },
        "1":{
            "name":"posterial",
            "data_number":0,
            "positive_number":0,
            "quality_level":{
                "0":{
                    "data_number":0,
                    "positive_number":0
                },
                "1":{
                    "data_number":0,
                    "positive_number":0
                },
                "2":{
                    "data_number":0,
                    "positive_number":0
                }
            }
        }
    }
}

# Load data from JSON file
with open(('../autodl-tmp/dataset_ROP/annotations.json')) as f:
    data_dict = json.load(f)
with open('../autodl-tmp/dataset_ROP/fid_annotations.json') as f:
    fid_annotations=json.load(f)
follow_up_set={k:0 for k in range(0,453)}
eye_set={k:0 for k in range(0,453)}
positive_follow_up_cnt=0
follow_up_number=0
postive_eye_cnt=0
for fid in fid_annotations:
    k = int(fid[:-1])
    follow_up_number+=1
    follow_up_set[k]+=1
    if fid_annotations[fid]>0:
        eye_set[k]=1
        positive_follow_up_cnt+=1
single_eye,both_eye=0,0
for eye in follow_up_set:
    if follow_up_set[eye]==2:
        both_eye+=1
    elif follow_up_set[eye]==1:
        single_eye+=1
    else:
        raise ValueError(f"no reocrd for eye id {eye}")
    
    if eye_set[eye]>0:
        postive_eye_cnt+=1
        
res["follow_up_number"]=follow_up_number
res["positve_follow_up"]=positive_follow_up_cnt
res["positive_eye"]=postive_eye_cnt
res["both_eye"]=both_eye
res["single_eye"]=single_eye

follow_up_split = {}
follow_up_image_cnt=[]
postive_eye_cnt_list=[]
negtive_eye_cnt_list=[]
for image_name in data_dict:
    fid = data_dict[image_name]['fid']
    if fid not in follow_up_split:
        follow_up_split[fid] = []
    follow_up_split[fid].append(image_name)
for fid in follow_up_split:
    follow_up_split[fid] = sorted(follow_up_split[fid], key=lambda x: int(x[:-4]))
    image_number=len(follow_up_split[fid])
    if image_number<=2:
        print(f"fid: {fid} have image number {image_number}")
    if image_number>=30:
        print(f"fid: {fid} have image number {image_number}")
    
    follow_up_image_cnt.append(len(follow_up_split[fid]))
    if fid_annotations[fid]>0:
        postive_eye_cnt_list.append(image_number)
    else:
        negtive_eye_cnt_list.append(image_number)
        
        
res["eye_min_number"]=min(follow_up_image_cnt)
res["eye_max_number"]=max(follow_up_image_cnt)
res["eye_average_number"]=(sum(follow_up_image_cnt))/len(follow_up_image_cnt)

# Count the data
for image_name in data_dict:
    data=data_dict[image_name]
    angle_type=str(data['angleType'])
    quality_level=str(data["qualityLevel"])
    
    res["angle_type"][angle_type]["data_number"]+=1
    res["angle_type"][angle_type]["quality_level"][quality_level]["data_number"]+=1
    if data['stage']>0:
        res["angle_type"][angle_type]["positive_number"]+=1
        res["angle_type"][angle_type]["quality_level"][quality_level]["positive_number"]+=1
    
    
save_path=os.path.join(save_dir,'base_condition.json') 
with open(save_path,'w') as f:
    json.dump(res,f,indent=4)

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


# 自定义颜色
colors = [(196/255, 216/255, 242/255), (165/255, 151/255, 182/255)]

# 字体设置
font_path = './arial.ttf'
font_size = 15
prop = FontProperties(fname=font_path, size=font_size)

# 创建图表
fig, ax = plt.subplots()

# 绘制箱线图
box1 = ax.boxplot(postive_eye_cnt_list, patch_artist=True, positions=[1], widths=0.6, boxprops=dict(facecolor=colors[0]))
box2 = ax.boxplot(negtive_eye_cnt_list, patch_artist=True, positions=[2], widths=0.6, boxprops=dict(facecolor=colors[1]))


# 设置轴标签
ax.set_xticklabels(['Positive', 'Negative'], fontproperties=prop)
ax.set_ylabel('Image Number', fontproperties=prop)

# 保存图像
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, 'eye_distribute.png'), bbox_inches='tight', dpi=300)

# 关闭图表以释放内存
plt.close()
