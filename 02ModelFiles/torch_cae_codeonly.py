import torch 
from torchvision.models import mobilenet_v2
import torch.nn.functional as F # for bce loss function
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import numpy as np
from typing import Tuple, List
from tqdm import tqdm
import torchvision.transforms as transforms

     
BATCH_SIZE = 32
INPUT_CHANNELS = 12
RES_BLOCK_INPUT_CHANNELS = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


plt.ion()
########## DATA PREP ############
"""Constants for the data reader."""

INPUT_FEATURES = ['elevation', 'th', 'vs',  'tmmn', 'tmmx', 'sph', 
                  'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']

OUTPUT_FEATURES = ['FireMask', ]

# Data statistics 
# For each variable, the statistics are ordered in the form:
# (min_clip, max_clip, mean, standard deviation)
DATA_STATS = {
    # Elevation in m.
    # 0.1 percentile, 99.9 percentile
    'elevation': (0.0, 3141.0, 657.3003, 649.0147),
    # Pressure
    # 0.1 percentile, 99.9 percentile
    'pdsi': (-6.12974870967865, 7.876040384292651, -0.0052714925, 2.6823447),
    'NDVI': (-9821.0, 9996.0, 5157.625, 2466.6677),  # min, max
    # Precipitation in mm.
    # Negative values do not make sense, so min is set to 0.
    # 0., 99.9 percentile
    'pr': (0.0, 44.53038024902344, 1.7398051, 4.482833),
    # Specific humidity.
    # Negative values do not make sense, so min is set to 0.
    # The range of specific humidity is up to 100% so max is 1.
    'sph': (0., 1., 0.0071658953, 0.0042835088),
    # Wind direction in degrees clockwise from north.
    # Thus min set to 0 and max set to 360.
    'th': (0., 360.0, 190.32976, 72.59854),
    # Min/max temperature in Kelvin.
    # -20 degree C, 99.9 percentile
    'tmmn': (253.15, 298.94891357421875, 281.08768, 8.982386),
    # -20 degree C, 99.9 percentile
    'tmmx': (253.15, 315.09228515625, 295.17383, 9.815496),
    # Wind speed in m/s.
    # Negative values do not make sense, given there is a wind direction.
    # 0., 99.9 percentile
    'vs': (0.0, 10.024310074806237, 3.8500874, 1.4109988),
    # NFDRS fire danger index energy release component expressed in BTU's per
    # square foot.
    # Negative values do not make sense. Thus min set to zero.
    # 0., 99.9 percentile
    'erc': (0.0, 106.24891662597656, 37.326267, 20.846027),
    # Population density
    # min, 99.9 percentile
    'population': (0., 2534.06298828125, 25.531384, 154.72331),
    # We don't want to normalize the FireMasks.
    # 1 indicates fire, 0 no fire, -1 unlabeled data
    'PrevFireMask': (-1., 1., 0., 1.),
    'FireMask': (-1., 1., 0., 1.)
}



"""
Data crop methods
Changes made:
- replaced tf.Tensor with torch.tensor in headers
- torch.cat
- replaced tf.image.random_crop with a RandomCrop transform 
- replaced tf.image.central_crop with a CentralCrop transform

"""


"""Randomly axis-align crop input and output image tensors.

Args:
    crop_img: tensor with dimensions HWC.
    output_img: tensor with dimensions HWC.
    desired_size: side length (square) to crop to.
    num_in_channels: number of channels in crop_img.
    num_out_channels: number of channels in output_img.
Returns:
    input_img: tensor with dimensions HWC.
    output_img: tensor with dimensions HWC.
"""
def random_crop_io(input_img: torch.tensor, output_img: torch.tensor,
    desired_size: int, num_in_channels: int, num_out_channels: int) -> Tuple[torch.tensor, torch.tensor]:

    # order should be CWH (no batch dim yet!!)
    # concat first (over CHANNELS, axis=1) so that image and label will be cropped the same, then separate again
    combined = torch.concatenate([input_img, output_img], axis=0) # CHW ordering (channel = axis0) rather than HWC as in tf
    cropTransform = transforms.RandomCrop(size=(desired_size, desired_size))
    combined = cropTransform(combined)
    
    # size CHW (channel first)
    input_img = combined[0:num_in_channels, :, :]
    output_img = combined[-num_out_channels:, :, :]

    return input_img, output_img

"""Center crops input and output image tensors.

Args:
    input_img: tensor with dimensions HWC.
    output_img: tensor with dimensions HWC.
    sample_size: side length (square) to crop to.
Returns:
    input_img: tensor with dimensions HWC.
    output_img: tensor with dimensions HWC.
"""
def center_crop_io(input_img: torch.tensor, output_img: torch.tensor,
    desired_size: int) -> Tuple[torch.tensor, torch.tensor]:

    # order is CHW not HWC as in torch so 0=channel, 1=height, 2=width
    central_fraction = desired_size / input_img.shape[1]
    cropH = int(input_img.shape[1]*central_fraction)
    cropW = int(input_img.shape[2]*central_fraction)

    cropTransform = transforms.CenterCrop(size=(cropH, cropW))
    input_img = cropTransform(input_img)
    output_img = cropTransform(output_img)

    return input_img, output_img


"""Clips and normalizes inputs with the stats corresponding to `key`.
Args:
    inputs: Inputs to clip and normalize.
    key: Key describing the inputs.

Returns:
    Clipped and normalized input.
"""
def clip_and_normalize(feature_name:str, inputs:torch.tensor):
    min_val, max_val, mean, std = DATA_STATS[feature_name]
    inputs = torch.clamp(input=inputs, min=min_val, max=max_val) # clip to specified range
    inputs = (inputs - mean)/std
    inputs = torch.nan_to_num(inputs, nan=0.0)

    return inputs


"""Clips and rescales inputs with the stats corresponding to `key`.
Args:
    inputs: Inputs to clip and rescale.
    feature_name: Key describing the inputs.

Returns:
    Clipped and rescaled input.
"""
def clip_and_rescale(feature_name:str, inputs:torch.tensor):
    min_val, max_val, _, _ = DATA_STATS[feature_name]
    inputs = torch.clamp(inputs=inputs, max=max_val, min=min_val)
    inputs = (inputs - min_val)/(max_val - min_val)
    inputs = torch.nan_to_num(inputs, nan=0.0)

    return inputs


######## CUSTOM DATASET ##############
class FireDataset(Dataset):

    def __init__(self, file_paths:List[str], data_size=64, crop_size=32, in_channels=12, 
                 clip_and_norm=True, clip_and_rescale=False, 
                 random_crop=True, center_crop=False):
        
        self.file_paths = file_paths
        self.center_crop = center_crop
        self.random_crop = random_crop
        self.clip_and_norm = clip_and_norm
        self.clip_and_rescale = clip_and_rescale
        self.data_size = data_size
        self.crop_size = crop_size
        self.in_channels = in_channels

        # read in all samples at once: leave preprocessing to getitem() method
        self.in_images = []
        self.labels = []
        self.base_dir = "C:\\Users\\rball\\OneDrive\\Documents\\school\\00 GRADUATE\\Thesis\\03DataVisualization\\mid-process-data-files\\csv-files\\"
        for file in self.file_paths:
            curr_file = self.base_dir + file
            df = pd.read_csv(curr_file)

            # can't just select the first column since they are not always in order
            target = df["FireMask"].astype(np.float32)
            target = target.values.reshape(64, 64, 1)

            data = df.drop('FireMask', axis=1).astype(np.float32)
            data = data[INPUT_FEATURES] # sort features into consistent order
            data = data.values.reshape(64, 64, 12)
            
            # permute to get in shape CHW. DONT unsqueeze for batch dim as that is added by DataLoader!!!!
            image = torch.tensor(data, dtype=torch.float32).permute(2,0,1)
            self.in_images.append(image)

            label = torch.tensor(target, dtype=torch.float32).permute(2,0,1)
            self.labels.append(label)
 
   
    def __len__(self):
        return len(self.file_paths)
    

    def __getitem__(self, idx):
        # in_images is already a list of tensors in the correct shape
        input_img = self.in_images[idx]
        output_img = self.labels[idx]

        if self.random_crop and self.center_crop:
            raise ValueError('Cannot have both random_crop and center_crop be True')

        if self.clip_and_norm:
            transformed = []
            for feature, channel in zip(INPUT_FEATURES, input_img.unbind(dim=0)):
                transformed.append(clip_and_normalize(feature, channel))
            input_img = torch.stack(transformed)
        elif self.clip_and_rescale:
            transformed = []
            for feature, channel in zip(INPUT_FEATURES, input_img.unbind(dim=0)):
                transformed.append(clip_and_rescale(feature, channel))
            input_img = torch.stack(transformed)

        if self.random_crop:
            input_img, output_img = random_crop_io(input_img, output_img, desired_size=self.crop_size, 
                                        num_in_channels=self.in_channels, num_out_channels=1)
        elif self.center_crop:
            input_img, output_img = center_crop_io(input_img, output_img, desired_size=self.crop_size)
            
        return input_img, output_img
        

test_file_paths = ["test_00.csv", "test_01.csv"]
eval_file_paths = ["eval_00.csv", "eval_01.csv"]
train_file_paths = ["train_00.csv", "train_01.csv", "train_02.csv", "train_03.csv", "train_04.csv", 
                    "train_05.csv", "train_06.csv", "train_07.csv", "train_08.csv", "train_09.csv", 
                    "train_10.csv", "train_11.csv", "train_12.csv", "train_13.csv", "train_14.csv"]

test_dataset = FireDataset(test_file_paths)
eval_dataset = FireDataset(eval_file_paths)
train_dataset = FireDataset(train_file_paths)


######### DATA VISUALIZATION #############
TITLES = [
  'Elevation',
  'Wind\ndirection',
  'Wind\nvelocity',
  'Min\ntemp',
  'Max\ntemp',
  'Humidity',
  'Precip',
  'Drought',
  'Vegetation',
  'Population\ndensity',
  'Energy\nrelease\ncomponent',
  'Previous\nfire\nmask',
  'Fire\nmask'
]


"""
Plot 'n_rows' rows of samples from dataset using a DataLoader.

Assumes inputs are in BCHW format and labels are in BCHW format with single channel.

Args:
    dataset (Dataset): PyTorch dataset.
    n_rows (int): Number of rows to plot.
    batch_size (int): Batch size for DataLoader. 
    """
def plot_samples_from_dataset(dataset: torch.utils.data.Dataset, n_rows: int): 
    global TITLES 

    fig = plt.figure(figsize=(15, 6.5))

    # Colormap for the fire masks
    CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
    BOUNDS = [-1, -0.1, 0.001, 1]
    NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)

    for i, (inputs, labels) in enumerate(dataset):
        if i >= n_rows:
            break
        
        n_features = inputs.shape[0]  # number of channels
        
        for j in range(n_features + 1):  # +1 for label column
            plt.subplot(n_rows, n_features + 1, i * (n_features + 1) + j + 1)
            
            # Title only for first row
            if i == 0:
                if j < n_features:
                    plt.title(TITLES[j], fontsize=13)
                else:
                    plt.title("Label", fontsize=13)
            
            # Input channels
            if j < n_features - 1:
                img = inputs[j, :, :].cpu().numpy()
                plt.imshow(img, cmap='viridis')
            
            # Second-to-last input channel and label
            if j >= n_features - 1:
                img = inputs[j, :, :].cpu().numpy() if j == n_features - 1 else labels[0, :, :].cpu().numpy()
                plt.imshow(img, cmap=CMAP, norm=NORM)
            
            plt.axis('off')

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

#plot_samples_from_dataset(train_dataset, 3)


####### EVALUATION METRICS ###########
"""
IoU metric
Calculation of intersection over union metric.
    
Args:
    real_mask (Tensor): Ground-truth mask
    predicted_mask (Tensor): Mask predicted by model
Returns:
    (float): IoU metric value

CHANGES: 
    - changed "tf" to "torch" in method header and real_mask line
    - added comments
"""
def IoU_metric(real_mask: torch.Tensor, predicted_mask: torch.Tensor) -> float: 
    # replacing neg values: torch.where(condition, choose-True, choose-False)
    # when the value is pos (>=0), keep the value from real_mastorch. otherwise, replace with 0
    real_mask = torch.where(real_mask>=0, real_mask, 0)

    # calculates the intersection and union between real and predicted by using a log AND and OR functions from numpy
    intersection = torch.logical_and(real_mask, predicted_mask)
    union = torch.logical_or(real_mask, predicted_mask)

    # if there is no object in either mask (both are entirely 0s), return 1 since IoU for 
    # empty masks would be perfect
    if torch.sum(union) == 0:
        return 1
    
    # else, calculate and return intersection over union (IoU)
    return torch.sum(intersection) / torch.sum(union)


"""
Calculation of recall metric.
    
Args:
    real_mask (Tensor): Ground-truth mask
    predicted_mask (Tensor): Mask predicted by model
Returns:
    (float): recall metric value

CHANGES:
    - changed tf to torch
"""
def recall_metric(real_mask: torch.Tensor, predicted_mask: torch.Tensor) -> float:

    real_mask = torch.where(real_mask < 0, 0, real_mask)
    
    true_positives = torch.sum(np.logical_and(real_mask, predicted_mask))
    actual_positives = torch.sum(real_mask)
    if actual_positives == 0:
        return 1
    
    return true_positives / actual_positives

"""
Calculation of precision metric.
    
Args:
    real_mask (Tensor): Ground-truth mask
    predicted_mask (Tensor): Mask predicted by model
Returns:
    (float): precision metric value

CHANGES:
    - changed tf to torch
"""
def precision_metric(real_mask: torch.Tensor, predicted_mask: torch.Tensor) -> float:
    real_mask = torch.where(real_mask < 0, 0, real_mask)
    
    true_positives = torch.sum(torch.logical_and(real_mask, predicted_mask))
    predicted_positives = torch.sum(predicted_mask)
    if predicted_positives == 0:
        return 1
    
    return true_positives / predicted_positives



########### LOSS FUNCTIONS ############
"""
Dice loss function calculator.
    
Args:
    y_true (Tensor): 
    y_pred (Tensor):
Returns:
    (Tensor): Dice loss for each element of a batch.

CHANGES:
    - changed tf to torch in method header
    - changed K to torch throughout
"""
def dice_coef(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    smooth = 1e-6
    # already flattened in calling function
    intersection = torch.sum(y_true * y_pred, axis=1)
    aplusb = (torch.sum(y_true, axis=1) + torch.sum(y_pred, axis=1) + smooth)
    dice = (2. * intersection + smooth) / aplusb
    return 1.0 - dice


"""
Calculates weighted binary cross entropy. The weights are fixed.
    
This can be useful for unbalanced catagories.

Adjust the weights here depending on what is required.

For example if there are 10x as many positive classes as negative classes,
    if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives 
    will be penalize 10 times as much as false negatives.

Args:
    true (Tensor): Ground-truth values
    pred (Tensor): Predited values
    weight_zero (float): Weight of class 0 (no-fire)
    weight_one (float): Weight of class 1 (fire)

Returns: 
    (float) : value for weighted binary cross entropy
CHANGES:
    - changed tf to torch in method header
    - changed K to torch throughout
    - changed keras BCE method to torch.nn.functional.binary_cross_entropy
    
"""
def weighted_bincrossentropy(true: torch.Tensor, pred: torch.Tensor, weight_zero: float = 0.01, weight_one: float = 1) -> float:
  
    # calculate the binary cross entropy
    # using torch.nn.functional.binary_cross_entropy, set reduction='none' to keep individual losses in a tensor
    # rather than taking mean 
    bin_crossentropy = F.binary_cross_entropy(input=pred, target=true, reduction='none')
    
    # apply the weights
    weights = true * weight_one + (1.0 - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy 
    
    return torch.mean(weighted_bin_crossentropy, axis=1)


"""
BCE loss function calculator.

Args:
    y_true (Tensor): 
    y_pred (Tensor):
Returns:
    (Tensor): Mean BCE Dice loss over a batch.

CHANGES:
    - changed tf to torch
    - originally returned keras.reduce_weighted_loss(loss), but without additional args all that did was 
      perform a sum operation. Replaced it with torch.sum since there is no torch equivalent to reduce_weighted_loss
"""
def bce_dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor):    
    y_true_f = torch.reshape(y_true, (BATCH_SIZE, -1))
    y_pred_f = torch.reshape(y_pred, (BATCH_SIZE, -1))

    bce = weighted_bincrossentropy(y_true_f, y_pred_f)
    dice = dice_coef(y_true_f, y_pred_f)

    sum = bce + dice

    # tf uses "reduce_weighted_loss" which basically takes the mean of a per-element loss vector
    return torch.mean(sum)



####### EVALUATION LOOP #############
from collections.abc import Callable
"""
Loads dataset according to file pattern and evaluates model's predictions on it.

Parameters:
    model (Callable[[tf.Tensor], tf.Tensor]): Function for model inference.
    eval_dataset (tf.dataDataset): Dataset for evaluation.

Returns:
    Tuple[float, float, float, float]: IoU score, recall score, precision score and mean loss.

CHANGES:
    - changed tf to torch
    - in method header, imported DataLoader from torch.utils and changed  eval_dataset: tf.data.Dataset) to DataLoader
    - changed tf.expand_dims(tf.cast(predictions, tf.float32), axis=-1) to predictions.float().unsqueeze(-1)
    in losses.append
"""
def evaluate_model(prediction_function: Callable[[torch.Tensor], torch.Tensor],
                   eval_dataset: DataLoader) -> Tuple[float, float, float, float]:
    IoU_measures = []
    recall_measures = []
    precision_measures = []
    losses = []
    
    for inputs, labels in tqdm(eval_dataset):
        # adding batch size first
        inputs = inputs.unsqueeze(0)
        labels = labels.unsqueeze(0)

        predictions = prediction_function(inputs)        

        for i in range(inputs.shape[0]):
            # dropping channels in NCHW
            IoU_measures.append(IoU_metric(labels[i, 0, :, :], predictions[i, :, :]))
            recall_measures.append(recall_metric(labels[i, 0, :, :], predictions[i, :, :]))
            precision_measures.append(precision_metric(labels[i, 0, :, :], predictions[i, :, :]))
        labels_cleared = torch.where(labels < 0, 0, labels)
        # add channel dimension back in so size is NCHW again
        losses.append(bce_dice_loss(labels_cleared, predictions.float().unsqueeze(1)
))
            
    mean_IoU = np.mean(IoU_measures)
    mean_recall = np.mean(recall_measures)
    mean_precision = np.mean(precision_measures)
    mean_loss = np.mean(losses)
    return mean_IoU, mean_recall, mean_precision, mean_loss



############# MODEL ################
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
        self.base_model.features[18].register_forward_hook(hook) # close to eqv 'block_16_project'

        
    def forward(self, x):
        self.base_model(x)
        return self.skips
    


    # to create upstack: replacing pix2pix.upsample layers
# pix2pix layers are just a conv transpose layer, a batchnorm layer, and a relu layer 
# (dropout optional but not used in NDWS model)
def upsample(input, output, kernel_size=3, stride=2, padding=1, output_padding=1):
    block = nn.Sequential(
        nn.ConvTranspose2d(in_channels=input, out_channels=output, kernel_size=kernel_size, 
                           stride=stride, padding=padding, output_padding=output_padding, bias=False),
        nn.BatchNorm2d(output),
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

        down_sizes = [576, 192, 144, 96]
        self.upstack = nn.Sequential(
                    upsample(1280, 512),                #1280,1,1 -> 512,2,2
                    upsample(512+down_sizes[0], 256),   #512+skip,2,2 -> 256,4,4
                    upsample(256+down_sizes[1], 128),   #256+skip,4,4 -> 128,8,8
                    upsample(128+down_sizes[2], 64)     #128+skip,8,8 -> 64,16,16
        )

    def forward(self, x):
        skips = self.downstack(x)
        x = skips[-1] # last layer of skips is bottleneck; where upsampler starts
        skips = reversed(skips[:-1]) # rearrange from deep->shallow, dropping the bottleneck
        
        # concatenate outputs of upstack and skips along channel dimension
        for up, skip in zip(self.upstack, skips):
            x = up(x)
            x = torch.cat([x, skip], dim = 1) # dim = 1 is channels acc to torch ordering

        convTranspose_last = nn.ConvTranspose2d(in_channels=x.shape[1], out_channels=1, kernel_size=3, stride=2, 
                                       padding=1, output_padding=1, bias=False)
        conv_last = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        sigmoid = nn.Sigmoid()
        
        x = convTranspose_last(x)
        x = conv_last(x)
        outputs = sigmoid(x)
            
        return outputs
    


    ######## TRAINING LOOP #################
    # FOR TROUBLESHOOTING ERROR
def plot_single_image(image: torch.Tensor, title):
    """
    Plot a single CHW image with variable channel count.
    If channels == 1 → title 'Firemap'
    Else → use TITLES for each channel.
    
    The last channel always uses the fire colormap.
    """

    global TITLES  # only used if channels > 1

    # Fire colormap setup: -1=black, 0=gray, 1=red
    CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
    BOUNDS = [-1, -0.1, 0.001, 1] 
    NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)

    # remove batch dim
    if image.dim() == 4:  # B, C, H, W
        image = image[0]  # take first sample

    C, H, W = image.shape

    # Choose number of columns
    n_cols = C

    plt.figure(figsize=(3 * n_cols, 4))

    for ch in range(C):
        plt.subplot(1, n_cols, ch + 1)

        img = image[ch].detach().cpu().numpy()

        # Title logic
        if C == 1:
            plt.title("Firemap", fontsize=13)
        else:
            plt.title(TITLES[ch], fontsize=13)

        # Last channel uses fire colormap
        if ch == C - 1:
            plt.imshow(img, cmap=CMAP, norm=NORM)
        else:
            plt.imshow(img, cmap="viridis")

        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.pause(0.001)



def train_model(model, train_data: DataLoader, epochs=10)-> Tuple[List[float], List[float]]:
    loss_fnc = bce_dice_loss
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    batch_losses = []
    eval_losses = []
    best_IOU = 0.0

    plt.ion()
    for epoch in range(epochs):
        losses = []
        print(f'Epoch {epoch+1}/{epochs}')

        # Iterate through the dataset
        # tqdm is a wrapper on the dataloader to display a progress bar
        progress = tqdm(train_data)

        for images, firemasks in progress:
            model.train()
            predictions = model(images)

            # converting "uncertain" (-1) labels to "no fire" (0)
            labels = torch.where(firemasks < 0, 0, firemasks) 

            # compute loss
            loss = loss_fnc(labels, predictions) # loss func returns a torch tensor
            losses.append(loss.detach().numpy()) # convert torch tensor to numpy array and append

            plot_single_image(images, "Input")
            plot_single_image(predictions, "Predictions")
            plot_single_image(labels, "Labels (Truth)")

            # display loss on tqdm progress bar
            progress.set_postfix({'batch_loss': loss.detach().numpy()})

            # Compute gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #END

        # Evaluating model:
        print("Evaluation...")
        model.eval()
        # lambda functions clips values to 1 if >0.5 and 0 otherwise (fire/no fire)
        IOU, recall, precision, val_loss = evaluate_model(
            lambda x: (model(x) > 0.5).int()[:,0,:,:], eval_dataset)
        
        print("Validation set metrics:")
        print(f"\tMean IoU: {IOU}\n\tMean precision: {precision}\n\tMean recall: {recall}\n\tValidation loss: {val_loss}")
        
        if IOU > best_IOU:
            best_IOU = IOU
            # they also saved the weights here but i skipped that for now

        # print losses
        print(f'Epoch: {epoch}, Train loss: {np.mean(losses)}\n')
        batch_losses.append(np.mean(losses))
        eval_losses.append(val_loss)
    #END

    print(f"Best model IoU: {best_IOU}")
    return batch_losses, eval_losses


# they set a seed for random here but not sure why?
        

def plot_train_and_val_losses(train_losses, val_losses):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(train_losses)
    axs[0].set_title("train loss")
    
    axs[1].plot(val_losses)
    axs[1].set_title("validation loss")
    
    plt.show(block=False)


model = convAutoencoder(INPUT_CHANNELS)
#test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
train_losses, eval_losses = train_model(model, train_loader, epochs=5)
plot_train_and_val_losses(train_losses, eval_losses) 

plt.show()