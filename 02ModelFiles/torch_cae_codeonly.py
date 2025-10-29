import torch
from torchvision.models import mobilenet_v2
import torch.nn.functional as F # for bce loss function
from torch.utils.data import DataLoader
import torch.nn as nn

#import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Tuple, List
from tqdm import tqdm

import pandas as pd
import numpy as np

     
BATCH_SIZE = 32
INPUT_CHANNELS = 12
RES_BLOCK_INPUT_CHANNELS = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# replacing: 
# tf.keras.applications.MobileNetV2(input_shape=[32, 32, 12], include_top=False, weights=None)
class Downstack(nn.Module):
    def __init__(self, input_channels):
        super().__init__() # initialize the class as a pytorch module

        self.base_model = mobilenet_v2(weights=None) # torch imp only takes weights as parameter
        # torch auto runs the classifier in the forward pass so have to replace those layers with identity to
        # get equivalent to include_top=False
        self.base_model.classifier = nn.Identity() 

        # changing default input size from 3 (RGB) to 12 for wildfire images
        self.base_model.features[0][0] = nn.Conv2d(
            in_channels = input_channels,
            out_channels = RES_BLOCK_INPUT_CHANNELS,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            bias = False
        )

        # replacing layer_names and base_model_outputs in keras
        # since we needed to grab outputs at submodule layers and only needed 5 rn, 
        # easiest way was to use a forward hook fnc and manually register each layer
        self.skips= []
        def hook(module, input, output):
            self.skips.append(output)

        self.base_model.features[2].conv[0].register_forward_hook(hook) # eqv 'block_1_expand_relu'
        self.base_model.features[3].conv[0].register_forward_hook(hook) # eqv 'block_3_expand_relu'
        self.base_model.features[6].conv[0].register_forward_hook(hook) # eqv 'block_6_expand_relu'
        self.base_model.features[13].conv[0].register_forward_hook(hook) # eqv 'block_13_expand_relu'
        self.base_model.features[18].register_forward_hook(hook) # eqv 'block_16_project'

        
    def forward(self, x):
        self.base_model(x)
        # for i in range(len(self.skips)):
        #     print(f"skips[{i}] shape: {self.skips[i].shape}")
        return self.skips
    

# to create upstack: replacing pix2pix.upsample layers
# pix2pix layers are just a conv transpose layer, a batchnorm layer, and a relu layer 
# (dropout optional but not used in NDWS model)
def upsample(input, output, kernel_size=3, stride=2, padding=1, out_pad=1):
    block = nn.Sequential(
        nn.ConvTranspose2d(in_channels=input, out_channels=output, kernel_size=kernel_size, 
                           stride=stride, padding=padding, output_padding=out_pad, bias=False),
        nn.BatchNorm2d(output),
        #nn.InstanceNorm2d(output),
        nn.ReLU()
    )
    nn.init.normal_(block[0].weight, mean=0.0, std=0.02) #initializer?
    return block    


# putting the downstack and upsampler together to make u-net like 
# conv autoencoder structure
class convAutoencoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__() # initialize as pytorch module

        self.downstack = Downstack(input_channels)

        self.upstack = nn.Sequential(
                    upsample(1280, 512),
                    upsample(512, 256),
                    upsample(256, 128),
                    upsample(128, 64)
                    # upsample(320, 576),
                    # upsample(576, 192),
                    # upsample(192, 144),
                    # upsample(144, 96)
        )

    def forward(self, x):
        skips = self.downstack(x)     

        x = skips[-1] # last layer of skips is bottleneck; where upsampler starts
        skips = reversed(skips[:-1]) # rearrange from deep->shallow, dropping the bottleneck

        # print("skips:")
        # for skip in skips:
        #     print(skip.shape)

        #print(self.upstack)
        
        # concatenate outputs of upstack and skips along channel dimension
        for up, skip in zip(self.upstack, skips):
            print(f"before up: {x.shape}, skip shape: {skip.shape}")
            x = up(x)
            print(f"after up: {x.shape}")
            #x = torch.cat([x, skip], dim = 1) # dim = 1 is channels acc to torch ordering
            
        return x
    
    
    

csv_file = "C:\\Users\\rball\\OneDrive\\Documents\\school\\00 GRADUATE\\Thesis\\03DataVisualization\\mid-process-data-files\\csv-files\\eval_00.csv"

df = pd.read_csv(csv_file)
target = df.iloc[:, 0].astype(np.float32)
data = df.iloc[:, 1:].astype(np.float32)

inputs = data.values.reshape(64, 64, 12)
labels = target.values.reshape(64, 64, 1)

inputs_crop = inputs[0:32, 0:32, :]
labels_crop = labels[0:32, 0:32, :]
print(inputs_crop.shape)
print(labels_crop.shape)

image_tensor = torch.tensor(inputs_crop, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
print(image_tensor.shape)

label_tensor = torch.tensor(labels_crop, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
print(label_tensor.shape)

model = convAutoencoder(12)
model.eval()
test = model(image_tensor)