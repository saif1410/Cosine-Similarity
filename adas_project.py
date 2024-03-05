# -*- coding: utf-8 -*-

#import required libraries 
import torch
#from yolov5 import utils
import torch
import utils
from IPython import display
from IPython.display import clear_output
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


from google.colab import drive  #connect drive with colab
drive.mount('/content/drive/')   #access the folder by connecting to google drive


%cd /content/drive/My Drive/Adas_object_detection/    # access the folder from collab interface

!ls      # the directory should have two files- datasets and yolov5 (clone from the yolov-5 repository into your folder structure)


%cd yolov5   # enter yolov-5 folder

!ls   # many files are present including the requirements.txt

%pip install -qr requirements.txt   # install the required packages 


!cat data/Adas_dataset.yaml   # yaml file has the details of the classes and details of train and validation data 

"""**TRAINING**"""

!python train.py --batch 32 --epochs 150 --data 'data/Adas_dataset.yaml' --weights 'yolov5x.pt' --project 'runs_Adas_dataset' --name 'feature_extraction' --cache --freeze 10

#!python train.py --resume --cache   # for later resuming training from saved weights (checkpoint)


display.Image(f"runs_Adas_dataset/feature_extraction/results.png")   #visualisation of training loss, precision and recall


!python train.py --hyp 'hyp.scratch-high.yaml' --batch 16 --epochs 100 --data 'data/Adas_dataset.yaml' --weights 'runs_Adas_dataset/feature_extraction/weights/best.pt' --project 'runs_Adas_dataset' --name 'fine-tuning' --cache    #fine-tuning post training

#!python train.py --resume --cache     # for later resuming fine-tuning from saved weights (checkpoint)

"""**VALIDATION**"""

!python val.py --weights 'runs_Adas_dataset/fine-tuning/weights/best.pt' --batch 64 --data 'data/Adas_dataset.yaml' --task val --project 'runs_Adas_dataset' --name 'validation_on_valid_data' --augment     # validation post fine-tuning

# precision recall curve for gauging the accuracy of the ADAS model 

plt.plot(figsize=(20,20))
plt.title('Precision Recall curve', fontsize=20)
plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
plt.imshow(mpimg.imread('runs_Adas_dataset/validation_on_valid_data/PR_curve.png'))

