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

os.makedirs(args.save_dir,exist_ok=True)
print("Saveing the model in {}".format(args.save_dir))
# Create the model and criterion
model= build_model(args.configs["model"],num_classes=2)# as we are loading the exite

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"using {device} for training")

# early stopping
early_stop_counter = 0

test_dataset=CustomDataset(
    split='test',data_path=args.data_path,split_name=args.split_name,resize=args.resize,data_angle_type=args.angle_type)

test_loader=  DataLoader(test_dataset,
                        batch_size=args.configs['train']['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])

# init metic

early_stop_counter = 0
best_val_loss = float('inf')
best_auc=0
save_model_name=f"{args.split_name}_{str(args.angle_type)}_{args.configs['save_name']}"

# Load the best model and evaluate
metirc=Metrics("Main")
model.load_state_dict(
        torch.load(os.path.join(args.save_dir, save_model_name)))
model.eval()
all_predictions = []
all_targets = []
all_probs = []

record_path=os.path.join(args.result_path,'pred_record',str(args.angle_type),args.configs['model']['name'])
parameter_key=f"{str(args.lr)}_{str(args.wd)}.json"
os.makedirs(record_path,exist_ok=True)

os.makedirs(f'./experiments/error/{args.angle_type}',exist_ok=True)
with open(os.path.join(args.data_path,'annotations.json')) as f:
    data_dict=json.load(f)


file_path=os.path.join(record_path,parameter_key)
if os.path.isfile(file_path):
    with open(file_path) as f:
        prediction_record=json.load(f)
else:
    prediction_record={}
    
with torch.no_grad():
    for inputs, targets, image_names in test_loader:
        inputs = to_device(inputs,device)
        targets = to_device(targets,device)
        outputs = model(inputs)
        probs = torch.softmax(outputs.cpu(), dim=1).numpy()
        predictions = np.argmax(probs, axis=1)
           
        for pred, label, image_name in zip(predictions, targets.cpu().numpy(), image_names):
            prediction_record[image_name]=int(pred)


with open(os.path.join(record_path,parameter_key),'w') as f:
    json.dump(prediction_record,f)

