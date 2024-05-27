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
model= build_model(args.configs["model"],num_classes=1)# as we are loading the exite


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
lr_scheduler=lr_sche(config=args.configs["lr_strategy"])
last_epoch = args.configs['train']['begin_epoch']

# Load the datasets
train_dataset=CustomDataset(
    split='train',data_path=args.data_path,split_name=args.split_name,resize=args.resize,data_angle_type=args.angle_type)
val_dataset=CustomDataset(
    split='val',data_path=args.data_path,split_name=args.split_name,resize=args.resize,data_angle_type=args.angle_type)
test_dataset=CustomDataset(
    split='test',data_path=args.data_path,split_name=args.split_name,resize=args.resize,data_angle_type=args.angle_type)
# Create the data loaders
    
train_loader = DataLoader(train_dataset, 
                          batch_size=args.configs['train']['batch_size'],
                          shuffle=True, num_workers=args.configs['num_works'],drop_last=True)
val_loader = DataLoader(val_dataset,
                        batch_size=args.configs['train']['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])
test_loader=  DataLoader(test_dataset,
                        batch_size=args.configs['train']['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])


from torch.nn import CrossEntropyLoss,MSELoss
# criterion = CrossEntropyLoss()
criterion = MSELoss()
if args.configs['model']['name']=='inceptionv3':
    from models import incetionV3_loss
    assert args.resize>=299, "for the model inceptionv3, you should set resolusion at least 299 but now "
    print('using inception loss')
    criterion= incetionV3_loss(args.smoothing)
# init metic
metirc= Metrics("Main")
print("batch size: {} batch size".format(args.configs["train"]['batch_size']))
print(f"Train: {len(train_dataset)} images in {len(train_loader)} batches")
print(f"Val: {len(val_dataset)} images in {len(val_loader)} batches")
print(f"Test: {len(test_dataset)} images in {len(test_loader)} batches")


early_stop_counter = 0
best_val_loss = float('inf')
best_acc=0
total_epoches=args.configs['train']['end_epoch']
save_model_name=f"{args.split_name}_{str(args.angle_type)}_{args.configs['save_name']}"
saved_epoch=-1
# Training and validation loop
for epoch in range(last_epoch,total_epoches):

    train_loss = train_epoch(model, optimizer, train_loader, criterion, device,lr_scheduler,epoch)
    val_loss,  metirc= val_epoch(model, val_loader, criterion, device,metirc)
    print(f"Epoch {epoch + 1}/{total_epoches}, "
      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
      )
    print(metirc)
    if metirc.accuracy>best_acc:
        best_acc= metirc.accuracy
        saved_epoch=epoch
        early_stop_counter = 0
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir,save_model_name))
        print("[Save Model In Epoch {}] Model saved as {}".format(str(epoch),os.path.join(args.save_dir,save_model_name)))
    else:
        early_stop_counter += 1
        if early_stop_counter > args.configs['train']['early_stop']:
            print("Early stopping triggered")
            break


# Load the best model and evaluate
metirc=Metrics("Main")
model.load_state_dict(
        torch.load(os.path.join(args.save_dir, save_model_name)))
val_loss, metirc=val_epoch(model, test_loader, criterion, device,metirc)
print(f"Best Epoch ")
print(metirc)
param={
    "model":args.configs["model"]["name"],
    "resolution": args.resize,
    "smoothing":args.smoothing,
    "split_name":args.split_name,
    "angle_type":args.angle_type,
    "optimizer":args.configs["lr_strategy"],
    "weight_decay":args.configs["train"]["wd"],
    "save_epoch":saved_epoch
}
metirc._store(saved_epoch,param)