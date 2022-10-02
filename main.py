from glob import glob

# from VertebraSegmentation.coordinate import *
# from VertebraSegmentation.filp_and_rotate import *
# from VertebraSegmentation.net.data import VertebraDataset
# from VertebraSegmentation.net.model.unet import Unet
# from VertebraSegmentation.net.model.resunet import ResidualUNet

from skimage import color, util, exposure
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_erosion, square
from skimage.filters import sobel,gaussian
from skimage.color import rgb2gray
from skimage import morphology
from skimage import io

import cv2
import math
import os 
import sys
import time
import datetime
import argparse
import shutil


from os.path import splitext, exists, join
from PIL import Image

import torch
import torch.nn as nn
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor
from torchvision import transforms
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from Landmark_Detection.model import DenseNet

#Spine Img Preprocessing - Contrast Adjustment 
def clahe_hist(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    return cl1

def Sort_coordinate(path, flag):
    # path = join("./coordinate", filename)
        
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    list_of_list = []
    
    for i, line in enumerate(lines):
        lines[i] = line.replace("\n", "")
        L = list(map(lambda x: int(x), lines[i].split(' ')))
        list_of_list.append(L)

    list_of_list.sort(key=lambda x: x[1])
    
    f1 = open(path, 'w')
    f2 = open(path, 'a')

    for i, line in enumerate(list_of_list):
        
        if flag:
            if i == 0:
                f1.write("{:d} {:d} {:d} {:d} {:d} {:d}\n".format(line[0], line[1], line[2], line[3], line[4], line[5]) )
                f1.close()
            else:
                f2.write("{:d} {:d} {:d} {:d} {:d} {:d}\n".format(line[0], line[1], line[2], line[3], line[4], line[5]) )
        else:
            if i == 0:
                f1.write("{:d} {:d} {:d} {:d}\n".format(line[0], line[1], line[2], line[3]) )
                f1.close()
            else:
                f2.write("{:d} {:d} {:d} {:d}\n".format(line[0], line[1], line[2], line[3]) )
    f2.close()
    
    return len(list_of_list)

def detect():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_folder", type=str, default="Dataset/test/images", help="path to dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, default="yolov5/checkpoints/best.pt", help="path to checkpoint model")
    
    opt = parser.parse_args()
    

    os.makedirs("./output", exist_ok=True)
    os.makedirs("./pre_img", exist_ok=True)
    os.makedirs("./coordinate", exist_ok=True)

    fname_list = []
    for file in os.listdir(opt.image_folder):
        
        file_name = splitext(file)[0]
        fname_list.append(f"{file_name}.txt")

    fname_list = sorted(fname_list)


    #Loads The Yolov5 Object Detection Model 
    model = torch.hub.load('yolov5','custom', path=opt.checkpoint_model, source='local')

    model.eval()

    #Load Tensor 
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    #Image received from the Server will change the code 
    img = cv2.imread('D:\Ally\Spine Images\i35.jpg',0) #Converts to Grayscale img

    #Adaptive Equalization - Contrasting
    img=clahe_hist(img) 

    #Resize Image Pixels  
    img = cv2.resize(img, (2057,6223),interpolation= cv2.INTER_NEAREST)
    

    detections = None

    with torch.no_grad():
        detections = model(img)


    plt.set_cmap('gray')
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)


    filename=("D:\Ally\Spine Images\i35.jpg")[-7:]

    results=detections.pandas().xyxy[0].to_dict(orient="records")
    if detections is not None:

        rewrite = True
        for result in results:
            con = result['confidence']
            cs = result['class']
            x1 = int(result['xmin'])
            y1 = int(result['ymin'])
            x2 = int(result['xmax'])
            y2 = int(result['ymax'])

            box_w = x2 - x1 
            box_h = y2 - y1 
            x1, y1, x2, y2 = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
            box_w, box_h = x2-x1, y2-y1

            if rewrite:
                f1 = open(f"./coordinate/{filename}.txt", 'w')
                f1.write("{:d} {:d} {:d} {:d} {:d} {:d}\n".format(x1, y1, x2, y2, box_w, box_h) )
                rewrite = False
            else:
                f1 = open(f"./coordinate/{filename}.txt", 'a')
                f1.write("{:d} {:d} {:d} {:d} {:d} {:d}\n".format(x1, y1, x2, y2, box_w, box_h) )
            
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=0.5, edgecolor='red', facecolor="none")
            ax.add_patch(bbox) 
    

    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    #filename = path.split("/")[-1].split(".")[0]
    #plt.savefig(f"output/{filename}.jpg", bbox_inches="tight", pad_inches=0.0, facecolor="none")
    plt.close()

    path1 = join("./coordinate", filename) 
    Sort_coordinate(f"{path1}.txt", flag=True)

    #detections.crop(save=True)  #Crop the Detected Vertebrae

    detections.print() #Print Results 
    detections.show() #Display Detected Vertebrae 
    detections.save() #Save Detected Vertebrae 
    
    # path2 = join("./GT_coordinate", filename)
    # Sort_coordinate(f"{path2}.txt", flag=False)
    # print('\n',detections.xyxy[0])  # img1 predictions (tensor)
    # print('\n',detections.pandas().xyxy[0]) # img1 predictions (pandas)


def main():
    detect()

if __name__ == "__main__":

    main()
