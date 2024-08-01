import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

data_path = '../quality_annotation'

class Visual_handler:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.data_path = '/mnt/d/dataset_ROP/images'
        font_path = './arial.ttf'
        # 加载字体，大小为40
        self.font = ImageFont.truetype(font_path, size=40)
        self.center = [800, 600]  # 修改为list

    def tuple_subtract(self, t1, t2):
        return [a - b for a, b in zip(t1, t2)]  # 修改为list

    def tuple_add(self, t1, t2):
        return [a + b for a, b in zip(t1, t2)]  # 修改为list

    def pos_legal(self, pos):
        x, y = pos
        return x >= 0 and x < 1600 and y >= 0 and y < 1200

    def legalize(self, pos):
        pos[0] = max(0, min(1530, pos[0]))
        pos[1] = max(0, min(1130, pos[1]))
        return pos

    def adjust_in(self, pos, mask):
        pos = self.legalize(pos)
        dis = np.linalg.norm(self.tuple_subtract(self.center, pos))
        if dis < 10:
            return pos
        dx = int(pos[0] / dis * 50)
        dy = int(pos[1] / dis * 50)
        while True:
            if mask[pos[1], pos[0]] == 1:
                return self.legalize(pos)
            pos[0] -= dx
            pos[1] -= dy
            if not self.pos_legal(pos):
                return self.legalize(pos)

    def adjust_out(self, pos, mask):
        pos = self.legalize(pos)
        dis = np.linalg.norm(self.tuple_subtract(self.center, pos))
        if dis < 10:
            return pos
        dx = int(pos[0] / dis * 50)
        dy = int(pos[1] / dis * 50)
        while True:
            if mask[pos[1], pos[0]] < 1:
                return self.legalize(pos)
            pos[0] += dx
            pos[1] += dy
            if not self.pos_legal(pos):
                return self.legalize(pos)

    def visual_posterior(self, image_name, save_path, peripheral_data, posterior_data):
        image_path = os.path.join(self.data_path, image_name)
        optic_disc_position = self.data_dict[image_name]['optic_disc_pred']['position']
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        mask = Image.open('mask.png').convert('L')
        mask = np.array(mask)
        mask[mask > 0] = 1
        # print(image_name)
        for data in posterior_data:
            rela_pos = self.tuple_subtract(self.center, data['position'])
            vis_pos = self.tuple_add(optic_disc_position, rela_pos)
            vis_pos = self.adjust_in(vis_pos, mask)
            # content = f"{data['id']}({str(data['quality_score'])})"
            content = f"{data['id']}"
            alpha = 0.1 + 0.9 * data['quality_score']
            color = (0, 0, 255, int(255 * alpha))  # 蓝色 with alpha
            draw.text(tuple(map(int, vis_pos)), content, font=self.font, fill=color)

        for data in peripheral_data:
            rela_pos = self.tuple_subtract(self.center, data['position'])
            vis_pos = self.tuple_add(optic_disc_position, rela_pos)
            vis_pos = self.adjust_out(vis_pos, mask)
            # print(vis_pos)
            # content = f"{data['id']}({str(data['quality_score'])})"
            content = f"{data['id']}"
            alpha = 0.1 + 0.9 * data['quality_score']
            color = (255, 255, 0, int(255 * alpha))  # 黄色 with alpha
            draw.text(tuple(map(int, vis_pos)), content, font=self.font, fill=color)

        quality_score = self.data_dict[image_name]['qualityScorePred']
        quality_score = round(quality_score, 2)
        draw.text((10, 10), f"Quality Score: {quality_score}", font=self.font, fill=(255, 255, 255))  # 白色文字

        # img = img.resize((400, 300), Image.Resampling.LANCZOS)  # 使用 LANCZOS
        img.save(save_path)

    def visual_peripheral(self, image_name, save_path):
        optic_disc_position = self.data_dict[image_name]['optic_disc_pred']['position']
        image_path = os.path.join(self.data_path, image_name)
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        quality_score = self.data_dict[image_name]['qualityScorePred']
        quality_score = round(quality_score, 2)
        # print(optic_disc_position)
        # 在 optic_disc_position 画一个黄色的实心圆，半径为10
        draw.ellipse([tuple(map(int, self.tuple_subtract(optic_disc_position, (10, 10)))), tuple(map(int, self.tuple_add(optic_disc_position, (10, 10))))], fill=(255, 255, 0))

        # 用白色在左上角写上 quality_score，字体采用上面加载的位置
        draw.text((10, 10), f"Quality Score: {quality_score}", font=self.font, fill=(255, 255, 255))

        img = img.resize((400, 300), Image.Resampling.LANCZOS)  # 使用 LANCZOS
        img.save(save_path)

    def is_posterior(self, image_name):
        return self.data_dict[image_name]['angleType_pred'] == 1

# 加载数据
with open(os.path.join(data_path, 'annotations.json')) as f:
    data_dict = json.load(f)
with open(os.path.join(data_path, 'fid_annotations.json')) as f:
    fid_annotations = json.load(f)

# 准备数据
follow_up_split = {}
for image_name in data_dict:
    fid = data_dict[image_name]['fid']
    if fid not in follow_up_split:
        follow_up_split[fid] = []
    follow_up_split[fid].append(image_name)
for fid in follow_up_split:
    follow_up_split[fid] = sorted(follow_up_split[fid], key=lambda x: int(x[:-4]))

with open(os.path.join(data_path, 'follow_up_split.json'), 'w') as f:
    json.dump(follow_up_split, f)

tar_path = '/mnt/d/feedbackBirthday'
os.system('rm -rf /mnt/d/feedbackBirthday/*')
os.makedirs(tar_path, exist_ok=True)
for condition in ['Positive', 'Negative']:
    os.makedirs(os.path.join(tar_path, condition), exist_ok=True)

visualer = Visual_handler(data_dict)
cnt = 100
for fid in fid_annotations:
    
    # fid='1d'
    
    condition = 'Positive' if fid_annotations[fid] > 0 else 'Negative'
    photo_number = len(follow_up_split[fid])
    fid_dir = os.path.join(tar_path, condition, fid + '_' + str(photo_number))
    os.makedirs(fid_dir)
    for angle_type in ['posterior', 'peripheral']:
        os.makedirs(os.path.join(fid_dir, angle_type))
    posterior_list = []
    peripheral_list = []
    for image_name in follow_up_split[fid]:
        quality_score = data_dict[image_name]['qualityScorePred']
        quality_score = round(quality_score, 2)
        data_record = {
            "image_name":image_name,
            "id": image_name.split('.')[0],
            "quality_score": quality_score,
            "position": data_dict[image_name]['optic_disc_pred']['position'],
        }
        if visualer.is_posterior(image_name):
            posterior_list.append(data_record)
        else:
            peripheral_list.append(data_record)
            visualer.visual_peripheral(image_name, os.path.join(fid_dir, 'peripheral', image_name))
    # print(peripheral_list)
    for data in posterior_list:
        visualer.visual_posterior(data["image_name"], os.path.join(fid_dir, 'posterior', data["image_name"]), 
                                  peripheral_data=peripheral_list,
                                  posterior_data=posterior_list)
    # raise
    cnt -= 1
    if cnt < 0:
        break
