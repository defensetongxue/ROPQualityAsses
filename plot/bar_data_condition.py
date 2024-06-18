import os
import json
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

save_dir = './experiments/record_figure/data_condition'
os.makedirs(save_dir, exist_ok=True)

# 数据示例
with open('./experiments/record_figure/data_condition/base_condition.json') as f:
    data = json.load(f)

# 自定义颜色
colors = {
    "angle_type": [[(196/255, 216/255, 242/255), (93/255, 116/255, 162/255)],
                   [(162/255, 151/255, 182/255), (69/255, 51/255, 112/255)]],
    "quality_level": [[(196/255, 216/255, 242/255), (93/255, 116/255, 162/255)],
                      [(162/255, 151/255, 182/255), (69/255, 51/255, 112/255)],
                      [(199/255, 160/255, 133/255), (123/255, 89/255, 94/255)]]
}

# 字体设置
font_path = './arial.ttf'
font_size = 15
prop = FontProperties(fname=font_path, size=font_size)

# 绘制堆积条形图
def plot_stacked_bar(data, categories, labels, colors, chart_name):
    fig, ax = plt.subplots(figsize=(6, 8))

    for i, (cat, color) in enumerate(zip(categories, colors)):
        data_number = data[cat]['data_number']
        positive_number = data[cat]['positive_number']
        ax.bar(labels[i], data_number - positive_number, color=color[0], width=0.5,label=f'{cat} - Negative')
        ax.bar(labels[i], positive_number, bottom=data_number - positive_number, width=0.5, color=color[1], label=f'{cat} - Positive')

    ax.set_xlabel('Sample Type', fontproperties=prop)
    ax.set_ylabel('Image Number', fontproperties=prop)
    ax.legend()

    plt.savefig(os.path.join(save_dir, chart_name), bbox_inches='tight', dpi=300)
    plt.close()

# 图1：不同角度类型中的正样本和数据总数
categories = ["1","0"]
labels = ["Posterial","Peripheral"]
chart_name = 'angle_type_distribution.png'
plot_stacked_bar(data["angle_type"], categories, labels, colors["angle_type"], chart_name)

# 图2：不同质量等级（角度类型0）中的正样本和数据总数
categories = ["0", "1", "2"]
labels = ["Quality 0", "Quality 1", "Quality 2"]
chart_name = 'peripheral_quality_distribution.png'
plot_stacked_bar(data["angle_type"]["0"]["quality_level"], categories, labels, colors["quality_level"], chart_name)

# 图3：不同质量等级（角度类型1）中的正样本和数据总数
chart_name = 'posterial_quality_distribution.png'
plot_stacked_bar(data["angle_type"]["1"]["quality_level"], categories, labels, colors["quality_level"], chart_name)
