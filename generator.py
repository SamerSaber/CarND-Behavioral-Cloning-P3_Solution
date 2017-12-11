import numpy as np
import cv2
import csv
from matplotlib import pyplot as plt
import random


def dataset_generator(batch_size=32, paths=[],  correction = 0.0):
    
    # Load Dataset Paths to memory to facilitate interaction between generator and dataset
    for path in paths:
        data_set = []
        with open(path) as log_csv:
            log_reader = csv.reader(log_csv)
            for line in log_reader:
                data_set.append((line[0],float(line[3])))
                data_set.append((line[1][1:],float(line[3]) + correction))
                data_set.append((line[2][1:],float(line[3]) - correction))
                
    while 1:
        
        
        
        j = 0
        i = 0
        images = []
        steering = []
        for data in data_set:
            
            
            try:
                
                if (abs(data[1]) <= 0.001):
                    j += 1
                    
                    if(j%30):
                        continue
                    
                    
                image = cv2.imread(data[0])
                #cv2.imshow('dataset',image)
                #cv2.waitKey(1)
                images.append(image)
                steering.append(data[1])
                if(i%2):
                    images.append(np.fliplr(image))
                    steering.append(data[1] * -1)

                i += 1                    
                if i == batch_size:
                    i = 0
                    X_train = np.array(images)
                    Y_train = np.array(steering)
                    images = []
                    steering = []
                    yield X_train, Y_train
                
        
            except:
                pass



# Generator Testing script

if __name__ == '__main__':

    data_generator = dataset_generator(batch_size=1000, paths=[r'./driving_log.csv'], correction=.15)
    for i in range(2):
        x, y = next(data_generator)
        print (i)
        print ('len(x) = '+str(len(x)))
