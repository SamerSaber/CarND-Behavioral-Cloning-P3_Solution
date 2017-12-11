import csv 
import cv2
import numpy as np
import matplotlib.pyplot as plt
def preprocess(paths):
    
    images = []
    steering_measurements = []
    lines = []
    correction = 0.15
    for path in paths: 
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
                
    
    
    i = 0
    for line in lines :
        
        if (abs(float(line[3])) <= 0.001):
            i += 1
            
            if(i%30):
                continue
            
            
        image = cv2.imread(line[0])
        #cv2.imshow('dataset',image)
        #cv2.waitKey(1)
        images.append(image)
        steering_measurements.append(float(line[3]))
        if(i%2):
            images.append(np.fliplr(image))
            steering_measurements.append(float(line[3]))
        
        
        images.append(cv2.imread(line[1][1:]))
        steering_measurements.append(float(line[3]) + correction)
        
        images.append(cv2.imread(line[2][1:]))
        steering_measurements.append(float(line[3]) - correction)
    X_train = np.array(images)
    Y_train = np.array(steering_measurements)
    
    plt.figure()
    plt.hist(Y_train, bins=100)
    plt.title("Training dataset Steering command Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show('Training dataset Steering command Histogram.png')
    return X_train, Y_train
    