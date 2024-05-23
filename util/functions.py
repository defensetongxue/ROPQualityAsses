import torch,math
from torch import optim
import numpy as np
from .metric import Metrics
from torch import nn
from collections import Counter
from .layer_decay import param_groups_lrd

def to_device(x, device):
    if isinstance(x, tuple):
        return tuple(to_device(xi, device) for xi in x)
    elif isinstance(x,list):
        return [to_device(xi,device) for xi in x]
    else:
        return x.to(device)

def train_epoch(model, optimizer, train_loader, loss_function, device,lr_scheduler,epoch):
    model.train()
    running_loss = 0.0
    batch_length=len(train_loader)
    for data_iter_step,(inputs, targets, meta) in enumerate(train_loader):
        # Moving inputs and targets to the correct device
        lr_scheduler.adjust_learning_rate(optimizer,epoch+(data_iter_step/batch_length))
        inputs = to_device(inputs, device)
        targets = to_device(targets, device)

        optimizer.zero_grad()

        # Assuming your model returns a tuple of outputs
        outputs = model(inputs)

        # Assuming your loss function can handle tuples of outputs and targets
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / len(train_loader)


def val_epoch(model, val_loader, loss_function, device,metirc:Metrics):
    loss_function=nn.CrossEntropyLoss()
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs = to_device(inputs,device)
            targets = to_device(targets,device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            running_loss += loss.item()
            probs = torch.softmax(outputs.cpu(), dim=1).numpy()
            predictions = np.argmax(probs, axis=1)
           

            all_predictions.extend(predictions)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs)
            
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probs = np.vstack(all_probs)
    # print(all_predictions.shape,all_probs.shape,)
    metirc.update(all_predictions,all_probs,all_targets)
    return running_loss / len(val_loader), metirc
def get_instance(module, class_name, *args, **kwargs):
    cls = getattr(module, class_name)
    instance = cls(*args, **kwargs)
    return instance

def get_optimizer(cfg, model):
    optimizer = None
    if cfg['model']['name']=='vit':
            return get_optimizer_vit(cfg,model) # layer decay is needed
    
    if cfg['train']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr'],
            momentum=cfg['train']['momentum'],
            weight_decay=cfg['train']['wd'],
            nesterov=cfg['train']['nesterov']
        )
    elif cfg['train']['optimizer'] == 'adam':
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr'],weight_decay=cfg['train']['wd']
        )
    elif cfg['train']['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr'],
            momentum=cfg['train']['momentum'],
            weight_decay=cfg['train']['wd'],
            alpha=cfg['train']['rmsprop_alpha'],
            centered=cfg['train']['rmsprop_centered']
        )
    else:
        raise
    return optimizer

class lr_sche():
    def __init__(self,config):
        self.warmup_epochs=config["warmup_epochs"]
        self.lr=config["lr"]
        self.min_lr=config["min_lr"]
        self.epochs=config['epochs']
    def adjust_learning_rate(self,optimizer, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs
        else:
            lr = self.min_lr + (self.lr  - self.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)))
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr
    
def get_optimizer_vit(cfg,model):
    param_groups = param_groups_lrd(model, cfg['train']['wd'],
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=cfg["train"]["layer_decay"]
    )
    optimizer = torch.optim.AdamW(param_groups, lr=cfg["train"]["lr"])
    return optimizer