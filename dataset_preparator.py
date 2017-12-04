import csv 
import cv2
import numpy as np

def preprocess(paths):
    
    images = []
    steering_measurements = []
    lines = []
    
    for path in paths: 
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
                
    
    i = 0
    number_of_flibed_data = 0
    for line in lines :
        image = cv2.imread(line[0])
        images.append(image)
        steering_measurements.append(float(line[3]))
        if (i % 2 == 0):
            images.append(np.fliplr(image))
            steering_measurements.append(float(line[3]) * -1)
            number_of_flibed_data += 1
        i += 1
        
    print("number_of_flibed_data = "+str(number_of_flibed_data))
    
    X_train = np.array(images)
    Y_train = np.array(steering_measurements)
    
    return X_train, Y_train
    