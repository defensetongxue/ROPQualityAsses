config_list = [
    "./configs/default.json",
    "./configs/inceptionv3.json",
    "./configs/resnet50.json",
    "./configs/vgg16.json",
    "./configs/efficientnet_b7.json",
]

split_list = ["1", "2", "3", "4"]
# lr_list = [1e-3, 1e-4]
# weight_decay = [5e-3, 5e-4]
lr_list = [1e-3]
weight_decay = [5e-4]
angle_types=[0,1]
with open('./todo.sh', 'w') as f:
    for angle in angle_types:
        for lr in lr_list:
            for wd in weight_decay:
                for cfg in config_list:
                    for split_name in split_list:
                        appdix_parameter = ""
                        if "inceptionv3" in cfg:
                            appdix_parameter = " --resize 299"
                        shell_line = f"python -u train.py --angle_type {angle} --cfg {cfg} --split_name {split_name} --lr {lr} --wd {wd}{appdix_parameter}\n"
                        f.write(shell_line)
