from torchvision import models
import os 
import torch.nn as nn
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
def build_inceptionv3(config,num_classes):
    os.environ['TORCH_HOME']=config["official_model_save"]
    model=models.inception_v3(pretrained=True)
    model.fc=nn.Linear(2048,num_classes)
    model.AuxLogits.fc=nn.Linear(768,num_classes)

    return model
def build_vgg16(config,num_classes):
    os.environ['TORCH_HOME'] = config["official_model_save"]
    model = models.vgg16(pretrained=True)
    # VGG16 has 4096 out_features in its last Linear layer
    model.classifier[6] = nn.Linear(4096,num_classes)

    return model
def build_resnet18(config,num_classes):
    os.environ['TORCH_HOME'] = config["official_model_save"]
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, num_classes)  # ResNet18 has 512 out_features in its last layer
    print(f"ResNet18 has {count_parameters(model)} parameters")
    return model

def build_resnet50(config,num_classes):
    os.environ['TORCH_HOME'] = config["official_model_save"]
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, num_classes)  # ResNet50 has 2048 out_features in its last layer
    print(f"ResNet50 has {count_parameters(model)} parameters")
    return model

def build_mobelnetv3_large(config,num_classes):
    os.environ['TORCH_HOME'] = config["official_model_save"]
    model=models.mobilenet_v3_large(pretrained=True)
    model.classifier[3]=nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    print(f"mobile net v3 large has {count_parameters(model)}")
    return model
def build_mobelnetv3_small(config,num_classes):
    os.environ['TORCH_HOME'] = config["official_model_save"]
    model=models.mobilenet_v3_small(pretrained=True)
    model.classifier[3]=nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    print(f"mobile net v3 large has {count_parameters(model)}")
    return model
def build_efficientnet_b7(config,num_classes):
    os.environ['TORCH_HOME'] = config["official_model_save"]
    model=models.efficientnet_b7(pretrained=True)
    model.classifier[1]=nn.Linear(in_features=2560, out_features=num_classes, bias=True)
    print(f"efficentnet b7 has {count_parameters(model)}")
    return model
def build_vit(config,num_classes):
    from .model_vit import build_vit
    model=build_vit(config["global_pool"],num_classes,config["pretrained"],config["drop_path"])
    return model
if __name__ =='__main__':
    cfg={
        "official_model_save":"./experiments",
        "num_classes":3
    }
    build_efficientnet_b7(cfg)