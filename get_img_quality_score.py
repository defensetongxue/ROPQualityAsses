import torch
from torch.utils.data import DataLoader
from util.dataset import CustomDataset
from  models import build_model
import os,json
import numpy as np
from util.metric import Metrics
from util.functions import to_device
from util.tools import visual
from configs import get_config
# Initialize the folder
os.makedirs("checkpoints",exist_ok=True)
os.makedirs("experiments",exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)
# Parse arguments
args = get_config()

# Create the model and criterion
model= build_model(args.configs["model"],num_classes=2)# as we are loading the exite

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"using {device} for training")


test_dataset=CustomDataset(
    split='test',data_path=args.data_path,split_name=args.split_name,resize=args.resize,data_angle_type=args.angle_type)

test_loader=  DataLoader(test_dataset,
                        batch_size=args.configs['train']['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])

with open(os.path.join(args.data_path,'annotations.json')) as f:
    data_dict=json.load(f)

prediction_record={}
with torch.no_grad():
    for inputs, targets, image_names in test_loader:
        inputs = to_device(inputs,device)
        targets = to_device(targets,device)
        outputs = model(inputs)
        probs = torch.softmax(outputs.cpu(), dim=1).numpy()
        predictions = np.argmax(probs, axis=1)
           
        for pred,prob, label, image_name in zip(predictions,probs, targets.cpu().numpy(), image_names):
            data_dict[image_name]['qualityScorePred']=float(round(prob[1],3))


with open(os.path.join(args.data_path,'annotations.json'),'w') as f:
    json.dump(data_dict,f)
    

