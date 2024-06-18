import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import sys
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from util.tools import get_color
from configs import get_config
args=get_config()
angle_type=str(args.angle_type)
# Set paths
record_path = f'./experiments/record_orignal/{angle_type}.json'
save_dir = f'./experiments/record_figure/ModelVsModel/{angle_type}'
font_path = './arial.ttf'
font_size = 15  # 增大字体大小
color_list = get_color("ganyu", 5)

# Ensure color values are normalized to [0, 1] for matplotlib
color_list = np.array(color_list) / 255.0

# Create directory to save results
os.makedirs(save_dir, exist_ok=True)

# Load local font
prop = FontProperties(fname=font_path, size=14)
y_tick_prop = FontProperties(fname=font_path, size=18)

# Load record file
with open(record_path) as f:
    records = json.load(f)

# Parse records and calculate mean and standard deviation
model_results = {}
split_results = {}
for record_item in records:
    result = record_item['result']
    params = record_item['param']
    model_name = params['model']
    lr = params['lr']
    wd = params['weight_decay']
    split_name = params['split_name']
    parameter_key = f"lr_{lr}_wd_{wd}"
    
    if model_name not in model_results:
        model_results[model_name] = {}
        split_results[model_name] = {}
    
    if parameter_key not in model_results[model_name]:
        model_results[model_name][parameter_key] = {
            'accuracy': [],
            'auc': [],
            'recall_0': [],
            'recall_1': [],
            'f1_0': [],
            'f1_1': []
        }
        split_results[model_name][parameter_key] = {
            'accuracy': {},
            'auc': {},
            'recall_0': {},
            'recall_1': {},
            'f1_0': {},
            'f1_1': {}
        }
    
    model_results[model_name][parameter_key]['accuracy'].append(result['accuracy'])
    model_results[model_name][parameter_key]['auc'].append(result['auc'])
    model_results[model_name][parameter_key]['recall_0'].append(result['recall'][0])
    model_results[model_name][parameter_key]['recall_1'].append(result['recall'][1])
    model_results[model_name][parameter_key]['f1_0'].append(result['f1'][0])
    model_results[model_name][parameter_key]['f1_1'].append(result['f1'][1])

    split_results[model_name][parameter_key]['accuracy'][split_name] = result['accuracy']
    split_results[model_name][parameter_key]['auc'][split_name] = result['auc']
    split_results[model_name][parameter_key]['recall_0'][split_name] = result['recall'][0]
    split_results[model_name][parameter_key]['recall_1'][split_name] = result['recall'][1]
    split_results[model_name][parameter_key]['f1_0'][split_name] = result['f1'][0]
    split_results[model_name][parameter_key]['f1_1'][split_name] = result['f1'][1]

# Calculate mean and standard deviation and find the best parameter key
summary_stats = {}
for model_name, param_results in model_results.items():
    summary_stats[model_name] = {}
    best_auc_mean = -1
    best_param_key = None

    for param_key, metrics in param_results.items():
        summary_stats[model_name][param_key] = {}
        for metric_name, values in metrics.items():
            mean_val = round(np.mean(values), 4)
            std_val = round(np.std(values), 4)
            summary_stats[model_name][param_key][metric_name] = [mean_val, std_val]

        # Check if this parameter key has the best AUC mean
        auc_mean = summary_stats[model_name][param_key]['auc'][0]
        if auc_mean > best_auc_mean:
            best_auc_mean = auc_mean
            best_param_key = param_key

    # Record the best parameter key's results
    best_results = summary_stats[model_name][best_param_key]
    lr, wd = best_param_key.split('_')[1], best_param_key.split('_')[3]
    summary_stats[model_name] = {
        "result": best_results,
        "best_params": {
            "lr": lr,
            "weight_decay": wd
        }
    }

# Save results to JSON file
result_json_path = os.path.join(save_dir, 'result.json')
with open(result_json_path, 'w') as f:
    json.dump(summary_stats, f, indent=4)

# Plotting function
def plot_metric(metric_name, summary_stats, split_results, save_dir, colors):
    models = list(summary_stats.keys())
    means = [summary_stats[model]['result'][metric_name][0] for model in models]
    stds = [summary_stats[model]['result'][metric_name][1] for model in models]  # Adjust the error bars

    fig, ax = plt.subplots(figsize=(5, 6))  # 增大图形尺寸
    bars = ax.bar(models, [(mean - 0.2) / 0.8 for mean in means], yerr=stds, capsize=5, color=colors[:len(models)], alpha=0.8)

    # Add bottom labels
    for bar, mean, color in zip(bars, means, colors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, 0, f'{mean:.3f}', ha='center', va='bottom', fontsize=font_size, color='black', fontproperties=prop)
    
    # Add scatter points for individual splits with grey border
    offsets = [0.2, 0.4, 0.6, 0.8]
    for i, model in enumerate(models):
        param_key = summary_stats[model]['best_params']
        lr_wd_key = f"lr_{param_key['lr']}_wd_{param_key['weight_decay']}"
        split_vals = list(split_results[model][lr_wd_key][metric_name].values())
        for j, split_val in enumerate(split_vals):
            ax.scatter(i + offsets[j % len(offsets)] - 0.5, (split_val - 0.2) / 0.8, color=colors[i], alpha=0.3, s=50, edgecolor='black')

    title = metric_name.replace('_', ' for class ').capitalize()
    ax.set_title(title, fontsize=font_size + 2, fontproperties=prop)
    ax.set_ylim(0, 1)  # Set y-axis limit from 0 to 1
    ax.set_yticks([0.25, 0.5, 0.75])  # Set y-axis ticks
    ax.set_yticklabels(['0.4', '0.6', '0.8'], fontsize=20, fontproperties=y_tick_prop)  # 增大字体大小
    # ax.set_yticklabels(['0.4', '0.6', '0.8'], fontsize=20)  # 增大字体大小
    ax.tick_params(axis='x', labelsize=font_size + 5)  # 增大x轴标签字体大小
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor', fontsize=font_size + 5, fontproperties=prop)  # 增大字体大小
    plt.tight_layout()  # 调整图形布局
    plt.savefig(os.path.join(save_dir, f'{metric_name}.png'), dpi=300)
    plt.close()

# Plot all metrics
for metric in ['accuracy', 'auc', 'recall_0', 'recall_1', 'f1_0', 'f1_1']:
    plot_metric(metric, summary_stats, split_results, save_dir, color_list)

# Create legend image
fig, ax = plt.subplots()
for i, color in enumerate(color_list[:len(model_results)]):
    ax.bar(0, 0, color=color, label=list(model_results.keys())[i])
ax.legend(loc='center', prop=prop)
ax.axis('off')
plt.savefig(os.path.join(save_dir, 'legend.png'), dpi=300)
plt.close()

print(f"Results and plots have been saved to {save_dir}")
