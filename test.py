import torch
from torch.utils.data import DataLoader
from util.dataset import CustomDataset
from  models import build_model
import os,json
import numpy as np
from util.metric import Metrics
from util.functions import get_optimizer,to_device
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
model= build_model(args.configs["model"],num_classes=3)# as we are loading the exite

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
with torch.no_grad():
    for inputs, targets, image_names in test_loader:
        inputs = to_device(inputs,device)
        targets = to_device(targets,device)
        outputs = model(inputs)
        probs = torch.softmax(outputs.cpu(), dim=1).numpy()
        predictions = np.argmax(probs, axis=1)
       
        all_predictions.extend(predictions)
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs)
        
all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)
all_probs = np.vstack(all_probs)

metirc.update(all_predictions,all_probs,all_targets)