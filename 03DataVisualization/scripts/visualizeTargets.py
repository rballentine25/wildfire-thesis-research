import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import numpy as np
import glob
import random
import os


def showTarget(img, title):
    img = img.to_numpy().reshape((64, 64))
    plt.imshow(img, cmap="gray_r", vmin=0, vmax=1)
    plt.title(title)
    #plt.axis('off')

    imgname = "03DataVisualization/target-images/" + title + ".png"
    plt.savefig(imgname)


def returnTarget(filename, inputFeatures, targName):
    df = pd.read_csv(filename)
    df = df[inputFeatures]
    img = df[targName]
    return img


INPUT_FEATURES = ['elevation', 'th', 'vs',  'tmmn', 'tmmx', 'sph', 
                  'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask', 'FireMask']
allfiles = os.listdir("03DataVisualization/renamed-raw-csvs/")
for file in allfiles:
    filepath = "03DataVisualization/renamed-raw-csvs/" + file
    img = returnTarget(filepath, INPUT_FEATURES, 'FireMask')
    showTarget(img, f"file{file[0:2]}")

