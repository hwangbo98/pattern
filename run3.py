from cProfile import label
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataset2 import *
from swin_t1 import *
import wandb
import argparse

parser = argparse.ArgumentParser(description="color & pattern training")

parser.add_argument("--pk_path", required=True, help='where is pickle path')
parser.add_argument("--device", default=0, help="device num")

args = parser.parse_args()

gpu_num = args.device
pk_path = args.pk_path

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num

# writer = SummaryWriter('runs/limeorange')
wandb.init(project="PatternClassification", entity="showniq_dll", config = {
  "learning_rate": 0.00001,
  "epochs": 0-3000,
  "batch_size": 32,
  "optimizer": "SGD",
  "momentum": 0.9,
  "learning_scheduler": "LambdaLR 0.85",
  "backbone" : "Swin_T",
  "pretrained" : "False",
  "Label Smoothing" : "False",
  "Fine tuning" : "True",
  "Cropped" : "True",
  "Weight decay" : 0.00001,
})
wandb.run.name = "expr2_pattern_cls_divide"
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainset = GenderImgDataset(pk_path, 'train')
validset = GenderImgDataset(pk_path, 'valid')
# validset = GenderImgDataset('/home/jewoo/face_blur_dataset/human/', 'valid')

trainloader = DataLoader(trainset, batch_size = 32, shuffle = True)
validloader = DataLoader(validset, batch_size = 32)
 
model = swin_t1(18, False)
model = model.to(device)
wandb.watch(model)

# checkpoint = torch.load('./model/genderbody/resnet101/best_valid_acc_model_expr0.pt', map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5, momentum=0.9, weight_decay=1e-5)

scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.85 ** epoch, last_epoch=-1, verbose=False)

best_train_loss = 500000
best_valid_loss = 500000
best_valid_acc = 0
num_samples = 0

for t in range(0, 3000):
    print(f"Epoch {t+1}\n------------------------")

    """Train code"""
    model.train()
    train_size = len(trainloader.dataset)
    # best_loss = ex_train_loss
    running_loss = 0
    running_correct = 0
    for batch, (X, y) in enumerate(trainloader):
        X, y = X.to(device), y.to(device)
        
        output = model(X)
        # _, pred = torch.max(output, 1)
        pred = torch.argmax(output, -1) #(N)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_samples += pred.shape[0]
        running_correct += torch.sum(pred == y.data)

        if batch != 0 and batch % 10 == 0:
            current = batch * len(X)
            if loss.item() < best_train_loss:
                best_train_loss = loss.item()
                torch.save({
                    'epoch' : t,
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'loss' : loss.item()
                }, "/mnt/hdd3/showniq/Data/pattern_cls_weights/swin_t" + "/best_train_model_" + wandb.run.name + ".pt")
            print(f"best loss: {best_train_loss:>7f} loss: {loss.item():>7f} running loss: {running_loss/10:>7f} [{current:>5d}/{train_size:>5d}]")
            wandb.log({
                "Train loss" : running_loss / 10,
                "Train accuracy" : running_correct.double() / num_samples
            })
            # writer.add_scalar("Loss train", running_loss/10, global_step=t*len(trainloader)+batch)
            # writer.add_scalar("Train Acc", running_correct.double() / (10 * 64), global_step=t*len(trainloader)+batch)
            running_loss = 0
            running_correct = 0
            num_samples = 0

    """ Valid code """
    val_size = len(validloader.dataset)
    num_batches = len(validloader)
    model.eval()
    valid_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in validloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            # _, pred = torch.max(output, 1)
            pred = torch.argmax(output, -1)
            valid_loss += criterion(output, y).item()
            correct += torch.sum(pred == y.data)
 
    valid_loss /= num_batches
    scheduler.step(valid_loss)    

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss

        torch.save({
                    'epoch' : t,
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'loss' : loss.item()
                },  "/mnt/hdd3/showniq/Data/pattern_cls_weights/swin_t" + "/best_valid_loss_model_" + wandb.run.name + ".pt")
        wandb.log({
            "Best Valid Epoch" : t
        })
    wandb.log({
        "Valid loss" : valid_loss,
        "Valid accuracy" : correct.double()/val_size
    })
    
    # writer.add_scalar("Loss valid", valid_loss, global_step=t*len(trainloader))
    # writer.add_scalar("correct", correct.double()/val_size, global_step=t*len(trainloader))

    
