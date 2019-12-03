import time
import os
import pandas as pd
import cv2 
import matplotlib.pyplot as plt
import json

IMAGE_BASE_PATH = "/home/malick/Bureau/home-assignment-0.1.0/assignment_imgs/"
BASE_PATH = "/home/malick/Bureau/home-assignment-0.1.0/"


with open(os.path.join(BASE_PATH, "img_annotations.json")) as f:
	annotations = json.load(f)  # get annotations


labels = pd.read_csv(os.path.join(BASE_PATH, "label_mapping.csv"))  #get labels
print(labels.head(10))

compt = 0
for image in annotations:
	compt += 1
	im = cv2.imread(os.path.join(IMAGE_BASE_PATH, image))
	for params in annotations[image]:
		x,y,h,w = params["box"][0] , params["box"][1], params["box"][2], params["box"][3]
		x_min = int(x)
		x_max = int(x + h)
		y_min = int(y)
		y_max = int(y + w)
		cv2.rectangle(im, (x_min,y_min),(x_max,y_max),(0,255,255),3)
		plt.imshow('bboxes',im)
		plt.show()
		cv2.waitKey(1000) #wait for 1s
	if(compt == 50) :
		break
