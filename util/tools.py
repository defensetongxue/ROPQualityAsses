from PIL import Image, ImageOps, ImageDraw, ImageFont
import os
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