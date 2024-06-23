from PIL import Image, ImageOps, ImageDraw, ImageFont
import os,json
import numpy as np
def visual(image_path, label, score, save_path, font_path='./arial.ttf', font_size=50):
    # 打开图像
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # 加载字体
    font = ImageFont.truetype(font_path, font_size)
    # 文本内容和位置

    text = f"Prediction: {score:.2f}, Label: {label}"
    text_position = (10, 10)  # 你可以根据需要调整文本位置
    
    # 文本颜色
    text_color = "white"
    
    # 在图像上绘制文本
    draw.text(text_position, text, fill=text_color, font=font)
    
    image=image.resize((800,600))
    # 保存图像
    image.save(save_path)
    
def get_color(color_name,number,file_path='./Color.json'):
    with open(file_path) as f:
        color_list=json.load(f)
    return color_list[color_name][str(number)]

from math import sin, cos, radians

def visual_optic_disc(image_path, position, save_path, save_resolu=(400, 300), circle_r=20):
    # 打开图像
    img = Image.open(image_path).convert('RGB')
    
    # 创建一个绘图对象
    draw = ImageDraw.Draw(img)
    
    # 计算圆圈的边界框
    x, y = position
    left_up_point = (x - circle_r, y - circle_r)
    right_down_point = (x + circle_r, y + circle_r)
    
    # 绘制蓝色圆圈
    draw.ellipse([left_up_point, right_down_point], outline='blue', width=3)
    
    # 计算检查点位置
    check_number = 48
    threshold = 50
    check_list = []
    angle_step = 360 / check_number
    for i in range(check_number):
        angle = radians(i * angle_step)
        check_list.append((int(threshold * sin(angle)), int(threshold * cos(angle))))
    
    # 绘制检查点
    for i, j in check_list:
        point_x, point_y = x + i, y + j
        draw.ellipse((point_x - 2, point_y - 2, point_x + 2, point_y + 2), fill='blue')
    
    # 调整图像大小
    img_resized = img.resize(save_resolu)
    
    # 保存图像
    img_resized.save(save_path)
    
    return
