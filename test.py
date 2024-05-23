import torch
from torch.utils.data import DataLoader
from util.dataset import CustomDataset
from  models import build_model
import os,json
import numpy as np
from util.metric import Metrics
from util.functions import train_epoch,val_epoch,get_optimizer,lr_sche
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
model= build_model(args.configs["model"])(# as we are loading the exite


# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"using {device} for training")

# early stopping
early_stop_counter = 0


# Creatr optimizer
model.train()
# Creatr optimizer
optimizer = get_optimizer(args.configs, model)

test_dataset=CustomDataset(
    split='test',data_path=args.data_path,split_name=args.split_name,resize=args.resize,data_angle_type=args.angle_type)
# Create the data loaders

test_loader=  DataLoader(test_dataset,
                        batch_size=args.configs['train']['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])

if args.smoothing> 0.:
    from timm.loss import LabelSmoothingCrossEntropy
    criterion =LabelSmoothingCrossEntropy(args.smoothing)
    print("Using tmii official optimizier")
else:
    from torch.nn import CrossEntropyLoss
    criterion = CrossEntropyLoss()
    criterion
if args.configs['model']['name']=='inceptionv3':
    from models import incetionV3_loss
    assert args.resize>=299, "for the model inceptionv3, you should set resolusion at least 299 but now "
    print('using inception loss')
    criterion= incetionV3_loss(args.smoothing)
# init metic
metirc= Metrics("Main")
print("There is {} batch size".format(args.configs["train"]['batch_size']))

early_stop_counter = 0
best_val_loss = float('inf')
best_auc=0
total_epoches=args.configs['train']['end_epoch']
save_model_name=f"{args.split_name}_{str(args.angle_type)}_{args.configs['save_name']}"

# Load the best model and evaluate
metirc=Metrics("Main")
model.load_state_dict(
        torch.load(os.path.join(args.save_dir, save_model_name)))
val_loss, metirc=val_epoch(model, test_loader, criterion, device,metirc)
