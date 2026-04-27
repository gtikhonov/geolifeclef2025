# In[1]:

import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
import glob

import pickle
import argparse
from typing import Optional
from functools import partial
import pandas as pd
import rasterio
# from flux_load_data_vInf3 import flux_dataset, flux_dataloader
# from flux_regress import RegressionModel_flux

from PIL import Image
# from flux_load_data_vInf3 import load_raster
# from flux_load_data_vInf3 import preprocess_image
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from torcheval.metrics import R2Score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torchgeo.trainers import BaseTask

from terratorch.models import EncoderDecoderFactory
from terratorch.datasets import HLSBands
from terratorch.tasks import PixelwiseRegressionTask
from terratorch.models.pixel_wise_model import freeze_module

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# In[2]:

class prithvi_terratorch(nn.Module):
  def __init__(self, prithvi_weight, model_instance, input_size):
    super(prithvi_terratorch, self).__init__()
    # load checkpoint for Prithvi_global
    self.weights_path = prithvi_weight
    self.checkpoint = torch.load(self.weights_path)
    self.input_size = input_size
    self.prithvi_model = model_instance   
    self.prithvi_model.load_state_dict(self.checkpoint, strict=False)

  def freeze_encoder(self):
    freeze_module(self.prithvi_model)

  def forward(self,x,temp,loc,mask):
    latent,_,ids_restore = self.prithvi_model.forward(x,temp,loc,mask)
    return latent
  
# In[3]:
  
def load_raster(path, if_img=1, crop=None):
  with rasterio.open(path) as src:
    img = src.read()
    # load  selected 4 bands for Sentinnel 2 (S2)
    if if_img==1:
      bands=[0,1,2,3]
      img = img[bands,:,:]
    # img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)# update our NO_DATA with -0.9999 -- chips are already scaled
    # print("img size",img.shape) 
    if crop:
      img = img[:, -crop[0]:, -crop[1]:]
  # print('return from load ras')
  return img


def preprocess_image(image,means,stds):        
  # normalize image
  means1 = means.reshape(-1,1,1)  # Mean across height and width, for each channel
  stds1 = stds.reshape(-1,1,1)    # Std deviation across height and width, for each channelzz
  normalized = image.copy()
  normalized = ((image - means1) / stds1)n
  normalized = torch.from_numpy(normalized.reshape(1, normalized.shape[0], 1, *normalized.shape[-2:])).to(torch.float32)
  #print('return from norm')
  return normalized
      
# In[3]  

class TrainDataset(Dataset):
  def __init__(self, dir_sentinel, dir_landsat, metadata, subset, num_classes, transform=None):
    self.subset = subset
    self.transform = transform
    self.dir_sentinel = dir_sentinel
    self.dir_landsat = dir_landsat
    self.num_classes = num_classes
    self.metadata = metadata
    self.metadata = self.metadata.dropna(subset="speciesId").reset_index(drop=True)
    self.metadata['speciesId'] = self.metadata['speciesId'].astype(int)
    self.label_dict = self.metadata.groupby('surveyId')['speciesId'].apply(list).to_dict()
    self.metadata = self.metadata.drop_duplicates(subset="surveyId").reset_index(drop=True)

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    survey_id = self.metadata.surveyId[idx]
    species_ids = self.label_dict.get(survey_id, [])  # Get list of species IDs for the survey ID
    label = torch.zeros(self.num_classes).scatter(0, torch.tensor(species_ids), torch.ones(len(species_ids)))

    sample_landsat = torch.nan_to_num(torch.load(os.path.join(self.dir_landsat, f"GLC25-PA-{self.subset}-landsat-time-series_{survey_id}_cube.pt"), weights_only=True))
    # Ensure the sample is in the correct format for the transform
    if isinstance(sample_landsat, torch.Tensor):
      sample_landsat = sample_landsat.permute(1, 2, 0)  # Change tensor shape from (C, H, W) to (H, W, C)
      sample_landsat = sample_landsat.numpy()  # Convert tensor to numpy array
    if self.transform:
      sample_landsat = self.transform(sample_landsat)
 
    dir1, dir2 = survey_id % 100, (survey_id // 100) % 100
    path_sentinel = os.path.join(self.dir_sentinel, f"{dir1}", f"{dir2}", f"{survey_id}.tiff")
    sample_sentinel = torch.nan_to_num(torch.tensor(load_raster(path_sentinel)))
    #TODO add transformations RandomHorizontalFlip, RandomRotation by 90 (write own)
    return sample_sentinel, sample_landsat, label, survey_id
    
class TestDataset(TrainDataset):
  def __init__(self, dir_sentinel, dir_landsat, metadata, subset, num_classes=None, transform=None):
    self.subset = subset
    self.transform = transform
    self.dir_sentinel = dir_sentinel
    self.dir_landsat = dir_landsat
    self.metadata = metadata
    self.num_classes = num_classes
      
  def __getitem__(self, idx):
    survey_id = self.metadata.surveyId[idx]
    sample_landsat = torch.nan_to_num(torch.load(os.path.join(self.dir_landsat, f"GLC25-PA-{self.subset}-landsat_time_series_{survey_id}_cube.pt"), weights_only=True))
    if isinstance(sample_landsat, torch.Tensor): #FIXME is this even needed in testing?
      sample_landsat = sample_landsat.permute(1, 2, 0)  # Change tensor shape from (C, H, W) to (H, W, C)
      sample_landsat = sample_landsat.numpy()
    if self.transform:
      sample_landsat = self.transform(sample_landsat)
      
    dir1, dir2 = survey_id % 100, (survey_id // 100) % 100
    path_sentinel = os.path.join(self.dir_sentinel, f"{dir1}", f"{dir2}", f"{survey_id}.tiff")
    sample_sentinel = torch.nan_to_num(torch.tensor(load_raster(path_sentinel)))
    return sample_sentinel, sample_landsat, survey_id
      

# In[3]
# Dataset and DataLoader
batch_size = 256
num_classes = 11255 
transform_landsat = transforms.Compose([
    transforms.ToTensor()
])

# Load Training metadata
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PATH, PRITHVI_WEIGHTS_PATH
path_data = DATA_PATH
train_path_sentinel = os.path.join(path_data, "SatelitePatches/PA-train")
train_path_landsat = os.path.join(path_data, "SateliteTimeSeries-Landsat/cubes/PA-train")
train_metadata_path = os.path.join(path_data, "GLC25_PA_metadata_train.csv")
train_metadata = pd.read_csv(train_metadata_path)
train_dataset = TrainDataset(train_path_sentinel, train_path_landsat, train_metadata, subset="train", num_classes=num_classes, transform=transform_landsat)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
print(train_dataset[0][0].shape)

# Load Test metadata
test_path_sentinel = os.path.join(path_data, "SatelitePatches/PA-test")
test_path_landsat = os.path.join(path_data, "SateliteTimeSeries-Landsat/cubes/PA-test")
test_metadata_path = os.path.join(path_data, "GLC25_PA_metadata_test.csv")
test_metadata = pd.read_csv(test_metadata_path)
test_dataset = TestDataset(test_path_sentinel, test_path_landsat, test_metadata, subset="test", transform=transform_landsat)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
print(test_dataset[0][0].shape)


      
# In[3]
from lightning import LightningModule
from terratorch.models.model import ModelOutput

#simple decoder to reduce dimensionality of prithvi enc output and flatten to 64D
class SimpleDecoder_comb_v2(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=64):
        super(SimpleDecoder_comb_v2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)# 1024 to 256; shape 10x1024 to 10x256
        #self.bn1 = nn.BatchNorm1d(hidden_dim)
        #self.drp = nn.Dropout(p=drp_rate)
        self.hidden_dim_flattened=10*hidden_dim#10 is feature dim+ class token in MAE; 10x256 to 2560
        self.fc2=nn.Linear(self.hidden_dim_flattened, output_dim)# 2560 to 64
        #self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        #self.gelu = nn.GELU()

    def forward(self, x):
        x = self.relu(self.fc1(x))#shape 10x1024 to 10x256 ORG
        x = torch.reshape(x,(x.shape[0], x.shape[1]*x.shape[2]))#10x256 to 2560 
        x = self.fc2(x)  # 2560 to 64 Output shape 
        return x

# Define the convolutional layers for the point1d MERRA input 
class Pt1dConvBranch(nn.Module):
    def __init__(self):
        super(Pt1dConvBranch, self).__init__()
        self.conv1 = nn.Conv2d(10, 32, kernel_size=1)
        #self.bn1 = nn.BatchNorm2d(32)
        #self.drp = nn.Dropout(p=drp_rate)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=1)
        #self.bn2 = nn.BatchNorm2d(16)
        #self.drp = nn.Dropout(p=drp_rate)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=1)
        self.fc = nn.Linear(8, 64)  # Final output matches decoder output

    def forward(self, x):
        x = torch.relu(self.conv1(x)) #ORIGINAL merra [batch_size, 10, 1, 1] to [batch_size, 32, 1, 1]
        x = torch.relu(self.conv2(x))## Output shape [batch_size, 16, 1, 1]
        x = torch.relu(self.conv3(x))#Output shape [batch_size, 8, 1, 1]
        x=torch.reshape(x, (x.shape[0], x.shape[1]))#output reshape [batch_size, 8]
        x = self.fc(x) # Output shape [batch_size, 64]
        return x      

# Define the regression model --simple regression to concatenate prithvi merra and regress to gpp lfux
class MultioutcomeModel(LightningModule):
  def __init__(self, prithvi_model):
    super(MultioutcomeModel, self).__init__()
    self.prithvi_model = prithvi_model
    self.decoder = SimpleDecoder_comb_v2(input_dim=1024, hidden_dim=256, output_dim=64)
    self.pt1d_conv_branch = Pt1dConvBranch()
    self.fc_final = nn.Linear(128, 1)  # Regression output
    #self.fc_final2 = nn.Linear(64, 1)  # Regression output

  def forward(self, im2d, pt1d, **kwargs):
    # Pass HLS im2d through the pretrained prithvi MAE encoder (with frozen weights)
    #pri_enc = self.prithvi_model(im2d, temporal_coords=None, location_coords=None)#.output#batch x 6x1x1x50; none, none for loc, temporal, 0--mask; output: batch x 10 x 1024
    pri_enc = self.prithvi_model(im2d, None, None, 0)#batch x 6x1x1x50; none, none for loc, temporal, 0--mask; output: batch x 10 x 1024

    # Pass pri_enc through the simple decoder
    dec_out = self.decoder(pri_enc)  # op Shape [batch_size, 64]
    # Pass MERRA pt1d through the convolutional layers
    pt1d_out = self.pt1d_conv_branch(pt1d)  # Shape [batch_size, 64]
    # Concatenate decoder output and pt1d output
    combined = torch.cat((dec_out[:, :], pt1d_out), dim=1) # op: [batch x 128]
    # Final regression output
    output1 = self.fc_final(combined)  # Shape [batch_size, 1]
    #output2 = self.fc_final2(output1)  # Shape [batch_size, 1]
    output = ModelOutput(output=output1)
    return output      

# In[3]

patch_size = [1,16,16]
n_frame = 1
n_channel = 4
embed_dim = 1024
decoder_depth = 8
num_heads = 16
mlp_ratio = 4
head_dropout = 0.0
      
# In[3]
# ### Creating an instance of our custom model used to estimate the carbon flux problem. 
path_prithvi = PRITHVI_WEIGHTS_PATH
wt_file = os.path.join(path_prithvi, "Prithvi_EO_V2_300M_TL.pt")

from huggingface_hub import hf_hub_download
from terratorch.models.backbones.prithvi_mae import PrithviViT
if not os.path.isfile(wt_file):
  hf_hub_download(repo_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL", filename="Prithvi_EO_V2_300M_TL.pt", local_dir=path_prithvi)

prithvi_instance = PrithviViT(
        patch_size=patch_size,
        num_frames=n_frame,
        in_chans=n_channel,
        embed_dim=embed_dim,
        decoder_depth=decoder_depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        head_dropout=head_dropout,
        backbone_input_size=[1,64,64],
        encoder_only=False,
        padding=True,
)
prithvi_model = prithvi_terratorch(wt_file, prithvi_instance, [1,64,64])
prithvi_model.freeze_encoder()

img0 = train_dataset[0][0]
img1 = train_dataset[0][0]
img_batch = torch.stack([img0, img1])[:,:,None,:,:] / 1000
prithvi_res = prithvi_model.forward(img_batch, None, None, 0)
prithvi_res

# from torchsummary import summary
# summary(prithvi_model, (4,1,64,64))








