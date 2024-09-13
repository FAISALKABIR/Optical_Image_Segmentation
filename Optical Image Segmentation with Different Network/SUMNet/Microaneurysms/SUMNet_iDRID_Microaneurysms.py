import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["CUDA_LAUNCH_BLOCKING"] = 1

import random
import numpy as np
import matplotlib.pyplot as plt

import json
import cv2
from tqdm import tqdm
from PIL import Image

PATH = "/home/faisal/6.IDRiD"
train_image_dir = os.path.join(PATH, "training/images")
train_ann_dir = os.path.join(PATH, "training/ground_truth/Microaneurysms")
val_image_dir = os.path.join(PATH, "validation/images")
val_ann_dir = os.path.join(PATH, "validation/ground_truth/Microaneurysms")
train_image_names = sorted(os.listdir(train_image_dir))
train_ann_names = sorted(os.listdir(train_ann_dir))
val_image_names = sorted(os.listdir(val_image_dir))
val_ann_names = sorted(os.listdir(val_ann_dir))



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF 
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SEGDataset(Dataset):
    def __init__(self, image_dir, ann_dir, image_names_list, ann_names_list, transform = None):
        super(SEGDataset, self).__init__()
        self.image_dir = image_dir
        self.ann_dir = ann_dir
        self.images = image_names_list
        self.anns = ann_names_list
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        ann_path = os.path.join(self.ann_dir, self.anns[index])
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #mask = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        mask = Image.open(ann_path)
        mask = np.array(mask)
        mask[mask > 1] = 1
        mask[mask < 1] = 0
        '''ann_file = open(ann_path)
        ann_data = json.load(ann_file)
        #tile_example(img_path, mask_rle, ex_id, organ, 
        mask = np.zeros(shape = image.shape[:-1])
        for idx in range(len(ann_data)):
            ann = [np.array(ann_data[idx], dtype = np.int32)]
            cv2.fillPoly(mask, pts = ann, color = (255, 255, 255))'''
        
        # mask[mask >= 128.0] = 1
        # mask[mask < 128.0] = 0
        
        if self.transform is not None:
            transformer = self.transform(image = image, mask = mask)
            image, mask = transformer["image"], transformer["mask"]
            
        return image, mask
    
#Splitting into train and val sets
random.seed(53)
np.random.seed(53)
torch.manual_seed(53)
'''np.random.shuffle(image_names)
N = len(image_names)
train_len = int(0.9 * N)
val_len = N - train_len'''
#train_image_names = image_names[:train_len]
#val_image_names = image_names[train_len:]
#print(f"No. of training images = {train_len}")
#print(f"No. of validation images = {val_len}")

def get_dataloaders(img_size = 256, batch_size = 16, image_dir = None, ann_dir = None, train_image_dir = None, val_ann_dir = None, train_image_names = None, train_ann_names = None, val_image_names = None, val_ann_names = None):
    
    
    train_transforms = A.Compose([
        A.PadIfNeeded(min_height=4288, min_width=4288, p=1),
        A.Resize(height = img_size, width = img_size, always_apply =True),
        A.HorizontalFlip(p = 0.5),
        A.VerticalFlip(p = 0.5),
        A.Rotate(limit = 90, p = 0.5),
        A.Normalize(
        mean = (0, 0, 0), std = (1.0, 1.0, 1.0), max_pixel_value = 255.0
        ),
        ToTensorV2()
    ])
    val_transforms = A.Compose([
        A.PadIfNeeded(min_height=4288, min_width=4288, border_mode=cv2.BORDER_CONSTANT, p=1),
        A.Resize(height = img_size, width = img_size, always_apply =True),
        A.Normalize(
        mean = (0, 0, 0), std = (1.0, 1.0, 1.0), max_pixel_value = 255.0
        ),
        ToTensorV2()
    ])
    

    train_dataset = SEGDataset(image_dir, ann_dir, train_image_names, train_ann_names, train_transforms)
    val_dataset = SEGDataset(val_image_dir, val_ann_dir, val_image_names, val_ann_names, val_transforms)

    g_seed = torch.Generator()
    g_seed.manual_seed(0)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 8)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = 8)
    
    return train_loader, val_loader

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms,datasets
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from focal_loss.focal_loss import FocalLoss
import torchvision
import sys
#from torchsummary import summary
from ptflops import get_model_complexity_info
from torchinfo import summary
from torchstat import stat


import matplotlib.pyplot as plt
import time
from torchvision import models

import torch
import torch.nn as nn
import torch.nn.functional as F


class SUMNet_all_bn(nn.Module):
	def __init__(self,in_ch,out_ch):
		super(SUMNet_all_bn, self).__init__()
		
		self.conv1     = nn.Conv2d(in_ch,64,3,padding=1)
		self.bn1       = nn.BatchNorm2d(64,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
		self.conv2     = nn.Conv2d(64,128,3,padding=1)   
		self.bn2       = nn.BatchNorm2d(128,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
		self.pool1     = nn.MaxPool2d(2, 2, return_indices = True)
		self.conv3a    = nn.Conv2d(128,256,3,padding=1) 
		self.bn3a       = nn.BatchNorm2d(256,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
		self.conv3b    = nn.Conv2d(256,256,3,padding=1)
		self.bn3b       = nn.BatchNorm2d(256,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
		self.pool2     = nn.MaxPool2d(2, 2, return_indices = True)
		self.conv4a    = nn.Conv2d(256,512,3,padding=1) 
		self.bn4a       = nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
		self.conv4b    = nn.Conv2d(512,512,3,padding=1)
		self.bn4b       = nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
		self.pool3     = nn.MaxPool2d(2, 2, return_indices = True)
		self.conv5a    = nn.Conv2d(512,512,3,padding=1) 
		self.bn5a       = nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
		self.conv5b    = nn.Conv2d(512,512,3,padding=1)
		self.bn5b       = nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
		self.pool4     = nn.MaxPool2d(2, 2, return_indices = True)  
		
		self.unpool4   = nn.MaxUnpool2d(2, 2)
		self.donv5b    = nn.Conv2d(1024, 512, 3, padding = 1)
		self.dbn5b     = nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
		self.donv5a    = nn.Conv2d(512, 512, 3, padding = 1)
		self.dbn5a     = nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
		self.unpool3   = nn.MaxUnpool2d(2, 2)
		self.donv4b    = nn.Conv2d(1024, 512, 3, padding = 1)
		self.dbn4b     = nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
		self.donv4a    = nn.Conv2d(512, 256, 3, padding = 1)
		self.dbn4a     = nn.BatchNorm2d(256,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
		self.unpool2   = nn.MaxUnpool2d(2, 2)
		self.donv3b    = nn.Conv2d(512, 256, 3, padding = 1)
		self.dbn3b     = nn.BatchNorm2d(256,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
		self.donv3a    = nn.Conv2d(256,128, 3, padding = 1)
		self.dbn3a     = nn.BatchNorm2d(128,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
		self.unpool1   = nn.MaxUnpool2d(2, 2)
		self.donv2     = nn.Conv2d(256, 64, 3, padding = 1)
		self.dbn2      = nn.BatchNorm2d(64,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
		self.donv1     = nn.Conv2d(128, 32, 3, padding = 1)
		self.dbn1       = nn.BatchNorm2d(32,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
		self.output    = nn.Conv2d(32, out_ch, 1)        
	   
		
	def forward(self, x):
		conv1          = F.relu(self.bn1(self.conv1(x)), inplace = True)
		conv2          = F.relu(self.bn2(self.conv2(conv1)), inplace = True)
		pool1, idxs1   = self.pool1(conv2)        
		conv3a         = F.relu(self.bn3a(self.conv3a(pool1)), inplace = True)
		conv3b         = F.relu(self.bn3b(self.conv3b(conv3a)), inplace = True)
		pool2, idxs2   = self.pool2(conv3b)
		conv4a         = F.relu(self.bn4a(self.conv4a(pool2)), inplace = True)
		conv4b         = F.relu(self.bn4b(self.conv4b(conv4a)), inplace = True)
		pool3, idxs3   = self.pool3(conv4b)
		conv5a         = F.relu(self.bn5a(self.conv5a(pool3)), inplace = True)
		conv5b         = F.relu(self.bn5b(self.conv5b(conv5a)), inplace = True)
		pool4, idxs4   = self.pool4(conv5b)
		
		unpool4        = torch.cat([self.unpool4(pool4, idxs4), conv5b], 1)
		donv5b         = F.relu(self.dbn5b(self.donv5b(unpool4)), inplace = True)
		donv5a         = F.relu(self.dbn5a(self.donv5a(donv5b)), inplace = True)
		unpool3        = torch.cat([self.unpool4(donv5a, idxs3), conv4b], 1)
		donv4b         = F.relu(self.dbn4b(self.donv4b(unpool3)), inplace = True)
		donv4a         = F.relu(self.dbn4a(self.donv4a(donv4b)), inplace = True)
		unpool2        = torch.cat([self.unpool3(donv4a, idxs2), conv3b], 1)
		donv3b         = F.relu(self.dbn3b(self.donv3b(unpool2)), inplace = True)
		donv3a         = F.relu(self.dbn3a(self.donv3a(donv3b)), inplace = True)
		unpool1        = torch.cat([self.unpool2(donv3a, idxs1), conv2], 1)
		donv2          = F.relu(self.dbn2(self.donv2(unpool1)), inplace = True)
		donv1          = F.relu(self.dbn1(self.donv1(torch.cat([donv2,conv1],1))), inplace = True)
		output         = self.output(donv1)          
	   
		return output           

def test_model():
    mod = SUMNet_all_bn(in_ch=3,out_ch=2)
    #x = torch.randn((1, 3, 512, 512))
    #mod.cuda()
    #pred = mod(x)
    #print(stat(mod, input_size=(3, 512, 512)))
    #print(summary(mod, input_size=(1, 3, 512, 512)))
    macs, params = get_model_complexity_info(mod, (3, 1024, 1024),
                                             as_strings=True,
                                             print_per_layer_stat=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    #print(pred.shape)

test_model()

def train(model, optimizer, loader, epoch, device = 'cpu'):
    model.train()
    loss_ep = 0
    dice_coeff_ep = 0
    jaccard_indx = 0
    f1_ep = 0
    for idx, (images, masks) in enumerate(tqdm(loader,  desc = f"EPOCH {epoch}")):
        images = images.to(device)
        masks = masks.to(device).long()
        #print(np.unique(masks.cpu().numpy()))
        # values, count = torch.unique(masks, return_counts = True)
        # #print(count)
        # alpha = count[1]/(count[0]+count[1])
        # alpha = torch.Tensor([alpha, 1 - alpha])
        # #masks = torch.unsqueeze(masks,1)
        # #print(masks.shape)
        outputs = model(images)
        #outputs = F.sigmoid(model(images))
        # outputs = outputs.view(outputs.size(0), outputs.size(1), -1)  # N,C,H,W => N,C,H*W
        # outputs = outputs.transpose(1, 2)                         # N,C,H*W => N,H*W,C
        # outputs = outputs.contiguous().view(-1, outputs.size(2))    # N,H*W,C => N*H*W,C
        # masks = masks.view(-1, 1)
        # logpt = F.log_softmax(outputs, dim=1)
        # logpt = logpt.gather(1,masks)
        # logpt = logpt.view(-1)
        # pt = logpt.exp()
        # if alpha.type() != outputs.data.type():
        #         alpha = alpha.type_as(outputs.data)
        # at = alpha.gather(0, masks.data.view(-1))
        # logpt = logpt * at
        # gamma = 2
        # loss = -1 * (1 - pt)**gamma * logpt
        # loss = loss.sum()
        #loss = criterion(outputs, masks.unsqueeze(1).float())
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_ep += loss.item()
        _, pred = outputs.max(1)
        #dice_coeff_ep += dice_coeff_binary(pred.detach(), masks.to(device).unsqueeze(1))
        dice_coeff_ep += dice(pred.detach(), masks.to(device).unsqueeze(1), average = 'weighted', num_classes = 2)
        jaccard_indx += ji(pred.detach(), masks.to(device), 'multiclass', num_classes=2)
        f1_ep += f1_score(pred.detach(), masks.to(device), 'multiclass', average = 'macro', multidim_average= 'global', num_classes = 2)
    train_loss = loss_ep/len(loader)
    train_dice_coeff = dice_coeff_ep/len(loader)
    train_jac_indx = jaccard_indx/len(loader)
    f1 = f1_ep/len(loader)
    return train_loss , train_dice_coeff, train_jac_indx, f1

def validate(model, loader, epoch, device = 'cpu'):
    model.eval()
    loss_ep = 0
    dice_coeff_ep = 0.0
    jaccard_indx = 0
    f1_ep = 0
    with torch.no_grad():
        for idx, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device).long()
            
            outputs = model(images)
            #outputs = F.sigmoid(model(images))
            #criterion = FocalLoss(gamma= 2, alpha = alpha)
            loss = criterion(outputs, masks)
            #loss = criterion(outputs, masks.unsqueeze(1).float())
            _, pred = outputs.max(1)
            #num_correct += (pred == masks).sum()
            #num_samples += pred.size(0)
            #dice_coeff_ep += dice_coeff_binary(pred.detach(), masks.to(device).unsqueeze(1))
            dice_coeff_ep += dice(pred.detach(), masks.to(device).unsqueeze(1), average = 'weighted', num_classes = 2)
            jaccard_indx += ji(pred.detach(), masks.to(device), 'multiclass', num_classes=2)
            f1_ep += f1_score(pred.detach(), masks.to(device), 'multiclass', average = 'macro', multidim_average= 'global', num_classes = 2)
            loss_ep += loss.item()
    val_loss = loss_ep/len(loader)
    dice_coeff = dice_coeff_ep/len(loader)
    jac_indx = jaccard_indx/len(loader)
    f1 = f1_ep/len(loader)
    return val_loss, dice_coeff, jac_indx , f1

# def dice_coeff_binary(pred_tensor, target_tensor):
#     pred = pred_tensor.flatten()
#     target = target_tensor.flatten()
    
#     intersection1 = torch.sum(pred * target)
#     intersection0 = torch.sum((1-pred) * (1-target))
    
#     coeff1 = (2.0*intersection1) / (torch.sum(pred) + torch.sum(target))
#     coeff0 = (2.0*intersection0) / (torch.sum(1-pred) + torch.sum(1-target))
    
#     return (coeff1+coeff0) / 2

def dice_coeff_binary(pred_tensor, target_tensor):
    #pred = pred_tensor.flatten()
    #target = target_tensor.flatten()
    
    intersection1 = torch.sum((pred_tensor * target_tensor).flatten())
    
    coeff1 = (2.0*intersection1) / (torch.sum(pred_tensor.flatten()) + torch.sum(target_tensor.flatten()))
    
    return coeff1

# a=sum((original_img*generated_img).flatten())
# b=(sum(generated_img.flatten()))+(sum(original_img.flatten()))


from torchmetrics.functional import jaccard_index as ji
from torchmetrics.functional import dice
from torchmetrics.functional import f1_score

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda'
img_size = 768
batch_size = 1

train_loader, val_loader = get_dataloaders(img_size, batch_size, train_image_dir, train_ann_dir, val_image_dir, val_ann_dir, train_image_names, train_ann_names, val_image_names, val_ann_names)

#model = MeDiAUNET(in_channels = 3, out_channels = 2 ,features = [64, 128, 256, 512])
#model = MeDiAUNET(in_channels = 3, out_channels = 2)
#model = SUMNet_all_bn(in_ch=3,out_ch=2)
model = SUMNet_all_bn(in_ch=3,out_ch=2)
model.to(device)

class FocalLoss(nn.Module):
    #WC: alpha is weighting factor. gamma is focusing parameter
    def __init__(self, gamma=0, alpha=None, size_average=True):
    #def __init__(self, gamma=2, alpha=0.25, size_average=False):    
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0002)
scheduler = ReduceLROnPlateau(optimizer, 'max',factor=0.5,patience=20,verbose = True, min_lr = 1e-6)

num_epochs = 400

save_checkpoint = True
checkpoint_freq = 5
load_from_checkpoint = False
load_pretrained = False
checkpoint_dir = "/home/faisal/SUMNet/Microaneurysms/weights"
save_dir = "/home/faisal/SUMNet/Microaneurysms/weights"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plot_history = True

model_path = "/home/faisal/SUMNet/Microaneurysms/weights"

class WrappedModel(nn.Module):
	def __init__(self, module):
		super(WrappedModel, self).__init__()
		self.module = module # that I actually define.
	def forward(self, x):
		return self.module(x)

# model = getattr(model, model)
# model = WrappedModel(model)

if load_from_checkpoint:
    checkpoint = torch.load(os.path.join(checkpoint_dir, "best_weight.tar"))
    model.load_state_dict(checkpoint["model_state_dict"])
    print('Model Loaded')
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

if load_pretrained:
    checkpoint = torch.load(os.path.join(model_path, "best_weight.tar"))
    model.load_state_dict(checkpoint["model_state_dict"])
    print('Model Loaded')
    #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    

train_loss_history, val_loss_history, train_dice_coeff_history, dice_coeff_history, train_jcrd_indx_history, jacrd_indx_history, train_f1_history, f1_history = [], [], [], [], [], [], [], []
best_dice_coeff = 0
train_loader_len  = len(train_loader)
val_loader_len = len(val_loader)

loop = tqdm(range(1, num_epochs + 1), leave = True)
for epoch in loop:
    train_loss, train_dice_coeff, train_jac_indx, train_f1 = train(model, optimizer, train_loader, epoch, device)
    val_loss, dice_coeff, jac_indx, f1 = validate(model, val_loader, epoch, device)
    #if(epoch > 20):
    scheduler.step(jac_indx)
    #scheduler.step()
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    train_dice_coeff_history.append(train_dice_coeff.cpu())
    dice_coeff_history.append(dice_coeff.cpu())
    train_jcrd_indx_history.append(train_jac_indx.cpu())
    jacrd_indx_history.append(jac_indx.cpu())
    train_f1_history.append(train_f1.cpu())
    f1_history.append(f1.cpu())
    
    print(f"Train loss = {train_loss} :: train_dice_coeff = {train_dice_coeff} ::train_jac_index = {train_jac_indx} :: train_f1_score = {train_f1} :: Val Loss = {val_loss} :: DICE Coeff = {dice_coeff.cpu()} :: Jaccard Index = {jac_indx} :: F1 Score = {f1}")
    
    if jac_indx > best_dice_coeff:
        torch.save({
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
        },os.path.join(checkpoint_dir,"best_weight.tar"))
        print('model saved')
        best_dice_coeff = jac_indx
    
    if save_checkpoint and epoch % checkpoint_freq == 0:
        torch.save({
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "epoch":epoch
        },os.path.join(save_dir,"checkpoint.tar"))
    
    
fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize = (30, 10))
ax[0].plot(range(1, num_epochs + 1), train_loss_history, label = "Train loss")
ax[0].plot(range(1, num_epochs + 1), val_loss_history, label = "Val loss")
ax[1].plot(range(1, num_epochs + 1), train_dice_coeff_history, label = "Train Dice Coefficient" )
ax[1].plot(range(1, num_epochs + 1), dice_coeff_history, label = "Dice Coefficient" )
ax[2].plot(range(1, num_epochs + 1), train_jcrd_indx_history, label = "Train Jaccard Index")
ax[2].plot(range(1, num_epochs + 1), jacrd_indx_history, label = "Jaccard Index")
ax[0].legend(fontsize = 20)
ax[1].legend(fontsize = 20)
ax[2].legend(fontsize = 20)
plt.savefig('/home/faisal/SUMNet/Microaneurysms/SUMNet_Microaneurysms.png')
plt.show()