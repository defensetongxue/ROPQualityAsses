
import os
from torchvision import  transforms
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import json
import torch

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
class CustomDataset(Dataset):
    def __init__(self, split,data_path,split_name,resize=224,data_angle_type=2):
        '''
        as the retfound model is pretrained in the image_net norm(mean,std),
        we keep the mean and std for this method, but for the other model, 
        '''
        with open(os.path.join(data_path,'split',f'{split_name}.json'), 'r') as f:
           split_list_all=json.load(f)[split]
        
        with open(os.path.join(data_path,'annotations.json'),'r') as f:
            self.data_dict=json.load(f)
        self.split_list=[]
        for image_name in split_list_all:
            if data_angle_type == 2 or \
                (data_angle_type ==1 == self.data_dict[image_name]['angleType']) :
                    self.split_list.append(image_name)
        self.preprocess=transforms.Compose([
            transforms.Resize((resize,resize))
        ])
        self.enhance_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])
        self.split = split
        
        self.img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD)])
        
    def __getitem__(self, idx):
        image_name = self.split_list[idx]
        data=self.data_dict[image_name]
        img=Image.open(data['image_path']).convert("RGB")
        img=self.preprocess(img)
        if self.split=='train':
            img=self.enhance_transforms(img)
            
        img=self.img_transforms(img)
        
        label = data['qualityLevel']
        return img,label,image_name


    def __len__(self):
        return len(self.split_list)
    
class Fix_RandomRotation(object):

    def __init__(self, degrees=360, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params():
        p = torch.rand(1)

        if p >= 0 and p < 0.25:
            angle = -180
        elif p >= 0.25 and p < 0.5:
            angle = -90
        elif p >= 0.5 and p < 0.75:
            angle = 90
        else:
            angle = 0
        return angle

    def __call__(self, img):
        angle = self.get_params()
        return F.rotate(img, angle, F.InterpolationMode.NEAREST , self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + \
            '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string
    
class CropPadding:
    def __init__(self, box=(80, 0, 1570, 1200)):
        self.box = box

    def __call__(self, img):
        return img.crop(self.box)