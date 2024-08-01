import torch
from torch.utils.data import DataLoader
from util.dataset import CustomDataset
from models import build_model
import os, json, random
import numpy as np
from util.metric import Metrics
from util.functions import train_epoch, val_epoch, get_optimizer, lr_sche
from configs import get_config

# Initialize the folder for saving checkpoints and experiments
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("experiments", exist_ok=True)

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

# Parse arguments from configuration
args = get_config()
args.configs["lr_strategy"]["lr"] = args.lr
args.configs["train"]["wd"] = args.wd
os.makedirs(args.save_dir, exist_ok=True)
print("Saving the model in {}".format(args.save_dir))

# Create the model and criterion
model = build_model(args.configs["model"], num_classes=2)  # Build the model with specified configuration

# Set up the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using {device} for training")

# Initialize early stopping counter
early_stop_counter = 0

# Set the model to training mode
model.train()

# Create optimizer and learning rate scheduler
optimizer = get_optimizer(args.configs, model)
lr_scheduler = lr_sche(config=args.configs["lr_strategy"])
last_epoch = args.configs['train']['begin_epoch']

# Load the datasets for training, validation, and testing
train_dataset = CustomDataset(
    split='train', data_path=args.data_path, split_name=args.split_name, resize=args.resize, data_angle_type=args.angle_type)
val_dataset = CustomDataset(
    split='val', data_path=args.data_path, split_name=args.split_name, resize=args.resize, data_angle_type=args.angle_type)
test_dataset = CustomDataset(
    split='test', data_path=args.data_path, split_name=args.split_name, resize=args.resize, data_angle_type=args.angle_type)

# Create data loaders for batching
train_loader = DataLoader(train_dataset, 
                          batch_size=args.configs['train']['batch_size'],
                          shuffle=True, num_workers=args.configs['num_works'], drop_last=True)
val_loader = DataLoader(val_dataset,
                        batch_size=args.configs['train']['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])
test_loader = DataLoader(test_dataset,
                         batch_size=args.configs['train']['batch_size'],
                         shuffle=False, num_workers=args.configs['num_works'])

# Set the loss function based on configuration
if args.smoothing > 0.:
    from timm.loss import LabelSmoothingCrossEntropy
    criterion = LabelSmoothingCrossEntropy(args.smoothing)
    print("Using timm official optimizer")
else:
    from torch.nn import CrossEntropyLoss, MSELoss
    criterion = CrossEntropyLoss()
    # criterion = MSELoss()

# Special handling for InceptionV3 model
if args.configs['model']['name'] == 'inceptionv3':
    from models import incetionV3_loss
    assert args.resize >= 299, "For the model InceptionV3, you should set resolution at least 299"
    print('Using InceptionV3 loss')
    criterion = incetionV3_loss(args.smoothing)

# Initialize metrics
metric = Metrics("Main")
print("Batch size: {} batch size".format(args.configs["train"]['batch_size']))
print(f"Train: {len(train_dataset)} images in {len(train_loader)} batches")
print(f"Val: {len(val_dataset)} images in {len(val_loader)} batches")
print(f"Test: {len(test_dataset)} images in {len(test_loader)} batches")

# Initialize early stopping parameters
early_stop_counter = 0
best_val_loss = float('inf')
best_auc = 0
total_epochs = args.configs['train']['end_epoch']
save_model_name = f"{args.split_name}_{str(args.angle_type)}_{args.configs['save_name']}"
saved_epoch = -1

# Training and validation loop
for epoch in range(last_epoch, total_epochs):
    train_loss = train_epoch(model, optimizer, train_loader, criterion, device, lr_scheduler, epoch)
    val_loss, metric = val_epoch(model, val_loader, criterion, device, metric)
    print(f"Epoch {epoch + 1}/{total_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    print(metric)

    # Save the model if the current AUC is the best
    if metric.auc > best_auc:
        best_auc = metric.auc
        saved_epoch = epoch
        early_stop_counter = 0
        torch.save(model.state_dict(), os.path.join(args.save_dir, save_model_name))
        print("[Save Model In Epoch {}] Model saved as {}".format(str(epoch), os.path.join(args.save_dir, save_model_name)))
    else:
        early_stop_counter += 1
        if early_stop_counter > args.configs['train']['early_stop']:
            print("Early stopping triggered")
            break

# Create directory for saving experiment records
os.makedirs(os.path.join('./experiments', 'record_original'), exist_ok=True)
save_path = os.path.join('./experiments', 'record_original', f"{str(args.angle_type)}.json")

# Load the best model and evaluate on the test set
metric = Metrics("Main")
model.load_state_dict(torch.load(os.path.join(args.save_dir, save_model_name)))
val_loss, metric = val_epoch(model, test_loader, criterion, device, metric)
print("Best Epoch")
print(metric)

# Save experiment parameters and metrics
param = {
    "model": args.configs["model"]["name"],
    "resolution": args.resize,
    "smoothing": args.smoothing,
    "split_name": args.split_name,
    "angle_type": args.angle_type,
    "optimizer": args.configs["lr_strategy"],
    "weight_decay": args.wd,
    "lr": args.lr,
    "save_epoch": saved_epoch
}
metric._store(saved_epoch, param, save_path)