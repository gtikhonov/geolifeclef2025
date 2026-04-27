import torch
import torch.nn as nn
import torchvision.models as tmodels
from terratorch.models.pixel_wise_model import freeze_module


class prithvi_terratorch(nn.Module):
  def __init__(self, prithvi_weight, model_instance, input_size):
    super(prithvi_terratorch, self).__init__()
    # load checkpoint for Prithvi_global
    self.weights_path = prithvi_weight
    self.input_size = input_size
    self.prithvi_model = model_instance
    if self.weights_path is not None:
      self.checkpoint = torch.load(self.weights_path)
      self.prithvi_model.load_state_dict(self.checkpoint, strict=False)

  def freeze_encoder(self):
    freeze_module(self.prithvi_model)

  def forward(self,x,temp,loc,mask):
    latent,_,ids_restore = self.prithvi_model.forward(x,temp,loc,mask)
    return latent


class ModifiedResNet18(nn.Module):
    def __init__(self, output_dim=1000):
        super(ModifiedResNet18, self).__init__()
        self.norm_input = nn.LayerNorm([8,4,18])
        self.resnet18 = tmodels.resnet18(weights=None)
        self.resnet18.conv1 = nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.maxpool = nn.Identity()
        # self.resnet18.layer4 = nn.Identity()
        # self.resnet18.fc = nn.Linear(256, output_dim)
        self.ln = nn.LayerNorm(1000)

    def forward(self, x):
        #x = self.norm_input(x)
        x = self.resnet18(x)
        #x = self.ln(x)
        return x

class SimpleDecoder(nn.Module):
    def __init__(self, input_dim=[17,1024], hidden_dim=256, output_dim=128):
        super(SimpleDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim[1], hidden_dim) # 1024 to 256; shape 17x1024 to 10x256
        self.hidden_dim_flattened=input_dim[0] * hidden_dim #17 is feature dim+ class token in MAE; 17x256 to 4352
        self.fc2 = nn.Linear(self.hidden_dim_flattened, output_dim) # 4352 to 128
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x)) #shape 17x1024 to 17x256 ORG
        x = torch.reshape(x,(x.shape[0], x.shape[1]*x.shape[2])) #17x256 to 4352 
        x = self.fc2(x) # 4352 to 128 Output shape 
        return x


class ModifiedPrithviResNet18(nn.Module):
    def __init__(self, num_classes, num_cov, resnet_dim, hidden_last_dim, prithvi_model):
        super(ModifiedPrithviResNet18, self).__init__()
        self.prithvi_model = prithvi_model
        self.decoder = SimpleDecoder(input_dim=[17,1024], hidden_dim=256, output_dim=128)
        self.landsat_part = ModifiedResNet18(resnet_dim)
        self.drop_tail = nn.Dropout(0.5)
        self.fc_tail = nn.Linear(128 + resnet_dim + num_cov, hidden_last_dim)
        self.relu_tail = nn.ReLU()
        self.drop_last = nn.Dropout(0.5)
        self.fc_final = nn.Linear(hidden_last_dim, num_classes)

    def forward(self, sentinel, landsat, cov, lonlat=None):
        x = self.prithvi_model(sentinel, None, lonlat, torch.tensor(0))
        x0 = self.decoder(x)
        x1 = self.landsat_part(landsat)
        x = torch.concat([x0, x1], -1)
        x = self.drop_tail(x)
        x = torch.concat([x, cov], -1)
        x = self.fc_tail(x)
        x = self.relu_tail(x)
        x = self.drop_last(x)
        x = self.fc_final(x)
        return x