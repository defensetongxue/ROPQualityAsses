import os
import json
import numpy as np
import matplotlib.pyplot as plt

# 设置路径
record_path = './record.json'
save_dir = './experiments/record_figure'
font_path = './arial.ttf'
font_size = 14

# 创建保存目录
os.makedirs(save_dir, exist_ok=True)

# 加载记录文件
with open(record_path) as f:
    records = json.load(f)

# 解析记录并计算均值和标准差
model_results = {}
for record_item in records:
    result = record_item['result']
    params = record_item['param']
    model_name = params['model']
    
    if model_name not in model_results:
        model_results[model_name] = {
            'accuracy': [],
            'auc': [],
            'recall_0': [],
            'recall_1': [],
            'recall_2': []
        }
    
    model_results[model_name]['accuracy'].append(result['accuracy'])
    model_results[model_name]['auc'].append(result['auc'])
    model_results[model_name]['recall_0'].append(result['recall'][0])
    model_results[model_name]['recall_1'].append(result['recall'][1])
    model_results[model_name]['recall_2'].append(result['recall'][2])

# 计算均值和标准差
summary_stats = {}
for model_name, metrics in model_results.items():
    summary_stats[model_name] = {}
    for metric_name, values in metrics.items():
        mean_val = round(np.mean(values), 4)
        std_val = round(np.std(values), 4)
        summary_stats[model_name][metric_name] = [mean_val, std_val]

# 保存结果到 JSON 文件
result_json_path = os.path.join(save_dir, 'result.json')
with open(result_json_path, 'w') as f:
    json.dump(summary_stats, f, indent=4)

# 绘制图表
def plot_metric(metric_name, summary_stats, save_dir):
    models = list(summary_stats.keys())
    means = [summary_stats[model][metric_name][0] for model in models]
    stds = [summary_stats[model][metric_name][1] for model in models]
    
    colors = plt.cm.tab20.colors  # 使用tab20颜色映射
    fig, ax = plt.subplots()
    bars = ax.bar(models, means, yerr=stds, capsize=5, color=colors[:len(models)], alpha=0.7)
    
    # 添加标准差的气泡标签和底部的白色数字
    for bar, mean, std, color in zip(bars, means, stds, colors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{std:.4f}', ha='center', va='bottom', fontsize=font_size, color=color)
        ax.text(bar.get_x() + bar.get_width() / 2.0, 0, f'{mean:.4f}', ha='center', va='bottom', fontsize=font_size, color='white', rotation=90)

    ax.set_xlabel('Model', fontsize=font_size)
    ax.set_ylabel(metric_name.capitalize(), fontsize=font_size)
    ax.set_title(f'{metric_name.capitalize()} by Model', fontsize=font_size)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{metric_name}.png'), dpi=300)
    plt.close()

# 绘制所有指标的图表
for metric in ['accuracy', 'auc', 'recall_0', 'recall_1', 'recall_2']:
    plot_metric(metric, summary_stats, save_dir)

print(f"Results and plots have been saved to {save_dir}")
