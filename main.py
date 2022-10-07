from skimage import color, util, exposure
from skimage.measure import label, regionprops
from skimage import morphology
from skimage import io

import cv2
import math
import os 
import sys
import time
import datetime
import argparse
from PIL import Image
import shutil

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from Landmark_Detection.model import DenseNet

parser = argparse.ArgumentParser()

parser.add_argument("--image_folder", type=str, default="Dataset/test/images", help="path to dataset")
parser.add_argument("--checkpoint_model", type=str, default="checkpoints/best.pt", help="path to checkpoint model")
parser.add_argument("--yolov5crop_folder", type=str, default="yolov5crop_folder/crop", help="path to patches folder")
parser.add_argument("--lndmarkdtct_folder", type=str, default="lndmarkdtct_folder/dtct", help="path to patches folder")

opt = parser.parse_args()

#Spine Img Preprocessing - Contrast Adjustment 
def clahe_hist(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    return cl1
    

def detect():

    os.makedirs("./yolov5crop_folder", exist_ok=True)
    os.makedirs("./lndmarkdtct_folder", exist_ok=True)

    #Loads The Yolov5 Object Detection Model 
    model = torch.hub.load('yolov5','custom', path=opt.checkpoint_model, source='local')

    #Loads The Landmark Detection Model 
    model2 = load_densenet_model("Landmark_Detection/outputs/model-306.h5")

    #Image received from the Server will change the code 
    img = cv2.imread('D:\Ally\Spine Images\Image_020_2ndbatch.jpg',0) #Converts to Grayscale img

    #Img Equalize     
    img = cv2.equalizeHist(img)

    #Img Histogram
    img=clahe_hist(img) 

    #Resize Image Pixels  
    img = cv2.resize(img, (1080,1920),interpolation= cv2.INTER_CUBIC)

    detections = None
    with torch.no_grad():
        detections = model(img)

    detections.print() #Print Results 
    # detections.show() #Display Detected Vertebrae 
    detections.save() #Save Detected Vertebrae 

    landmarks=[]
    cnt=0
    results=detections.pandas().xyxy[0].to_dict(orient="records")
    if detections is not None:
        rewrite = True
        for result in results:
            # con = result['confidence']
            # cs = result['class']
            x1 = int(result['xmin'])
            y1 = int(result['ymin'])
            x2 = int(result['xmax'])
            y2 = int(result['ymax'])
            
            cnt=cnt+1
            print("Landmark ",cnt)
            print("x-min|y-min|x-max|y-max")
            print(x1,y1,x2,y2)
            # crop the patch from the original image
            patch = img[y1:y2, x1:x2]
            cv2.imwrite((opt.yolov5crop_folder)+str(cnt)+'.jpg',patch)
            
            #Show Cropped Images 
            # cv2.imshow("Cropped", patch)
            # cv2.waitKey(0)

            # predict local landmark within the patch
            local_landmark = predict_landmark(patch, model2,cnt)
            
            # convert the landmark's local position to the global position with respect to original image    
            landmark_global_pos = np.zeros_like(local_landmark)
            landmark_global_pos[0:8:2] = (local_landmark[0:8:2] + x1)/img.shape[1]
            # print("Width",img.shape[1])
            # print(local_landmark[0:8:2] + x1)
            # print((local_landmark[0:8:2] + x1)/img.shape[1])
            landmark_global_pos[1:8:2] = (local_landmark[1:8:2] + y1)/img.shape[0]
            # print("Height",img.shape[0])
            # print(local_landmark[1:8:2] + y1)
            # print((local_landmark[1:8:2] + y1)/img.shape[0])

            landmarks.append(landmark_global_pos)

        #change the order of coordinates within list; all x coordinates first followed by y coordinates 
        landmarks = np.array(landmarks).ravel().tolist()
        landmarks = landmarks[::2] + landmarks[1::2]

        print(int(len(landmarks)))

        save_landmarks_image(img,landmarks,'D:/Ally/CADMA/Landmark_Outputs/finale.jpg') 


        # avg_slopes=[]
        # for m in range (0,int(len(landmarks)/2),4):
        #     slope1 = (round(img.shape[0]*landmarks[m+1+int(len(landmarks)/2)])-round(img.shape[0]*landmarks[m+int(len(landmarks)/2)]))/(
        #              round(img.shape[1]*landmarks[m+1])-round(img.shape[1]*landmarks[m]))
        #     slope2 = (round(img.shape[0]*landmarks[m+3+int(len(landmarks)/2)])-round(img.shape[0]*landmarks[m+2+int(len(landmarks)/2)]))/(
        #             round(img.shape[1]*landmarks[m+3])-round(img.shape[1]*landmarks[m+2]))
        #     avg_slopes.append((slope1+slope2)/2)
        
        # cobb_angles= find_cobb_angles(avg_slopes,cnt)
        
        # print("Computed Cobb Angles")
        # print (cobb_angles)

        slopes =[]
        avg_slopes=[]
        lines = []
        cobb_angles= [0.0,0.0,0.0]

        for m in range (0,int(len(landmarks)/2),2):
            slope = (round(img.shape[0]*landmarks[m+1+int(len(landmarks)/2)])-round(img.shape[0]*landmarks[m+int(len(landmarks)/2)]))/(round(img.shape[1]*landmarks[m+1])-round(img.shape[1]*landmarks[m]))
            slopes.append(slope)
            lines.append(((int(img.shape[1]*landmarks[m+1]),int(img.shape[0]*landmarks[m+1+int(len(landmarks)/2)])),(int(img.shape[1]*landmarks[m]),int(img.shape[0]*landmarks[m+int(len(landmarks)/2)]))))

        for s in range (0,int((len(landmarks)/2)/2),2):
            avg_slopes.append((slopes[s]+slopes[s+1])/2)
        
            
        avg_slope = np.array(avg_slopes)
        max_slope = np.amax(avg_slopes)
        min_slope= np.amin(avg_slopes)
        if np.argmax(avg_slopes)> np.argmin(avg_slopes):
            lower_MT= np.argmax(avg_slopes)
            upper_MT = np.argmin(avg_slopes)
        else:
            upper_MT= np.argmax(avg_slopes)
            lower_MT = np.argmin(avg_slopes)

        #upper_MT= np.argmax(avg_slopes)
        #lower_MT = np.argmin(avg_slopes)    
        print (avg_slopes)
        print ("Maximum Tilted vertebra",upper_MT,lower_MT)
        print (" Their Slopes: ", avg_slopes[0:upper_MT+1])

            
        upper_max_slope= np.amax(avg_slopes[0:upper_MT+1])
        upper_min_slope = np.amin(avg_slopes[0:upper_MT+1])


        lower_max_slope=np.amax(avg_slopes[lower_MT:cnt])
        lower_min_slope=np.amin(avg_slopes[lower_MT:cnt])

        print ("Upper max-min slopes: ", upper_max_slope, upper_min_slope)
        print ("Lower max-min slopes: ",lower_max_slope, lower_min_slope)
        print (np.rad2deg(np.arctan(upper_max_slope)), np.rad2deg(np.arctan(upper_min_slope)))

        cobb_angles[0]= abs(np.rad2deg(np.arctan(max_slope))- np.rad2deg(np.arctan(min_slope)))
        cobb_angles[1]= abs(np.rad2deg(np.arctan(upper_max_slope))-np.rad2deg(np.arctan(upper_min_slope)))
        cobb_angles[2]= abs(np.rad2deg(np.arctan(lower_max_slope)) - np.rad2deg(np.arctan(lower_min_slope)))

        [cv2.line (img,pt[0],pt[1],(255,0,0),5) for pt in lines]

        cv2.imwrite('D:/Ally/CADMA/Output.jpg',img)

        print (" Cobb angles", cobb_angles)

        ScolioClass = ''
        for i in range(3):
            print(cobb_angles[i])
            if cobb_angles[i] >= 10 and cobb_angles[i] <= 24:
               print("mild")
               ScolioClass = 'Mild Scoliosis'
            elif cobb_angles[i] >= 25 and cobb_angles[i] <= 50:
                print("moderate")
                ScolioClass = 'Moderate Scoliosis'
            elif cobb_angles[i] > 50:
                print("severe")
                ScolioClass = 'Severe Scoliosis'
        print(ScolioClass)

def predict_landmark(image, model,ctr):

    # im = cv2.resize(image,(200,120),interpolation=cv2.INTER_NEAREST)
    im = image
    # im = np.delete(im, [1, 2], axis=2)
    im = np.array(im) / 255.0
    im = np.expand_dims(im, axis=0)
    print(im.shape)
    lmarks = model.predict(im)
    lmarks = lmarks[0]
    print(lmarks)

    lmarks[0:8:2] = lmarks[0:8:2] * im.shape[2]
    lmarks[1:8:2] = lmarks[1:8:2] * im.shape[1]
    im = im[0] * 255
    # im = np.squeeze(im, axis=(2,))
    for m in range(0, 8,2):
        cv2.circle(im, (int(lmarks[m]), int(lmarks[m + 1])), 5,(0,255,0), -1)
    
    plt.figure(1, figsize=(25, 25))
    plt.subplot(211)
    plt.imshow(im, cmap=cm.gray)
    cv2.imwrite((opt.lndmarkdtct_folder)+str(ctr)+'.jpg',im)

    return np.around(lmarks)

def load_densenet_model(model_path):
    model = DenseNet(dense_blocks=5, dense_layers=-1, growth_rate=8, dropout_rate=0.2, bottleneck=True, compression=1.0, weight_decay=1e-4, depth=40)
    model.load_weights(model_path)
    return model

def save_landmarks_image(img, landmark, img_output_path):

    for m in range(0, int(len(landmark)/2)):
        cv2.circle(img, (int(img.shape[1]*landmark[m]), int(img.shape[0]*landmark[m + int(len(landmark)/2 )])), 5, (0, 0, 255), -1)

    cv2.imwrite(img_output_path ,img)
   
def find_cobb_angles(vertebra_slopes,ctr):
    
    cobb_angles= [0.0,0.0,0.0]
    if not isinstance(vertebra_slopes, np.ndarray):
        vertebra_slopes= np.array(vertebra_slopes)
    
    max_slope = np.amax(vertebra_slopes)
    min_slope= np.amin(vertebra_slopes)
    
    lower_MT= np.argmax(vertebra_slopes)
    upper_MT = np.argmin(vertebra_slopes)
    
    if lower_MT < upper_MT:
        lower_MT, upper_MT = upper_MT, lower_MT
    

    upper_max_slope= np.amax(vertebra_slopes[0:upper_MT+1])
    upper_min_slope = np.amin(vertebra_slopes[0:upper_MT+1])

    lower_max_slope=np.amax(vertebra_slopes[lower_MT:ctr])
    lower_min_slope=np.amin(vertebra_slopes[lower_MT:ctr])


    print ("Maximum Tilted vertebra",upper_MT,lower_MT)

    cobb_angles[0]= abs(np.rad2deg(np.arctan(max_slope))- np.rad2deg(np.arctan(min_slope)))

    cobb_angles[1]= abs(np.rad2deg(np.arctan(upper_max_slope))-np.rad2deg(np.arctan(upper_min_slope)))

    cobb_angles[2]= abs(np.rad2deg(np.arctan(lower_max_slope)) - np.rad2deg(np.arctan(lower_min_slope)))
    
    return cobb_angles

def main():
    detect()

if __name__ == "__main__":

    main()
