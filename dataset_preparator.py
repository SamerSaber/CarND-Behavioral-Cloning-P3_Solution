import random
import csv 
import cv2
import numpy as np
import matplotlib.pyplot as plt
def preprocess(paths):
    
    images = []
    steering_measurements = []
    lines = []
    correction = .15
    for path in paths: 
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
                
    
    
    i = 0
    j = 0
    REGION_04_1 = 0
    REGION_15_17 = 0
    REGION_11_18 = 0
    REGION_1_12 = 0
    REGION_12_165 = 0
    REGION_01_034 = 0
    for line in lines :
        j += 1
        if (abs(float(line[3])) <= 0.0001):
            i += 1
            
            if(i%24):
                continue
             
        
        
        elif (abs(float(line[3])) > 0.135 and abs(float(line[3])) < 0.18):
            REGION_15_17 +=1 
            if (REGION_15_17 % 4):
                continue
#         
#         elif (abs(float(line[3])) > 0.12 and abs(float(line[3])) < 0.165):
#             REGION_12_165 +=1 
#             if (REGION_12_165 % 8):
#                 continue
# 
#         elif (abs(float(line[3])) > 0.11 and abs(float(line[3])) < 0.18):
#             REGION_11_18 +=1 
#             if (REGION_11_18 % 6):
#                 continue
# 
#         elif (abs(float(line[3])) > 0.1 and abs(float(line[3])) < 0.165):
#             REGION_1_12 +=1 
#             if (REGION_1_12 % 2):
#                 continue
#             
#         elif (abs(float(line[3])) > 0.01 and abs(float(line[3])) < 0.035):
#             REGION_01_034 +=1 
#             if (REGION_01_034 % 2):
#                 continue    
    
        center_image = cv2.imread(line[0])
        left_image   = cv2.imread(line[1][1:])
        right_image  = cv2.imread(line[2][1:])
        
#         center_image = cv2.resize(center_image,(320,160), interpolation = cv2.INTER_CUBIC)
#         left_image   = cv2.resize(center_image,(320,160), interpolation = cv2.INTER_CUBIC)
#         right_image  = cv2.resize(center_image,(320,160), interpolation = cv2.INTER_CUBIC)
        
        images.append(center_image)
        steering_measurements.append(float(line[3]))
        

        images.append(left_image)
        steering_measurements.append(float(line[3]) + correction)
         
        images.append(right_image)
        steering_measurements.append(float(line[3]) - correction)
        
        
        images.append(np.fliplr(center_image))
        steering_measurements.append(float(line[3]) * -1)

        images.append(np.fliplr(left_image))
        steering_measurements.append(float(line[3]) * -1 - correction)

        images.append(np.fliplr(right_image))
        steering_measurements.append(float(line[3]) * -1 + correction)

        
#         cv2.imshow('dataset_c_c',cv2.imread(line[0])[70:140, 0:320])
#         cv2.imshow('dataset_l_c',np.fliplr(cv2.imread(line[2][1:])[70:140, 0:320]))
#         cv2.imshow('dataset_r_c',cv2.imread(line[2][1:])[60:140, 0:320])
#         
#         cv2.imshow('dataset_c',center_image)
#         cv2.imshow('dataset_l',left_image)
#         cv2.imshow('dataset_r',right_image)
#          
#         cv2.waitKey(0)

    X_train = np.array(images)
    Y_train = np.array(steering_measurements)
    
    plt.figure()
    plt.hist(Y_train, bins=50)
    plt.title("Training dataset Steering command Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(r'./data_histogram.png')
    plt.show('Training dataset Steering command Histogram.png')
    return X_train, Y_train


def filter(X = [], Y = [] , number_of_items = 500, number_of_pins = 20):
    

    dataset = list(zip(X, Y))
    
    random.shuffle(dataset)
    
    X_filtered, Y_filtered = zip(*dataset)
    X_filtered = list(X_filtered)
    Y_filtered = list(Y_filtered)
    
    
    max_val = max(Y_filtered)
    
    offset = max_val/number_of_pins
    #print ("offset = "+ str(offset))
    number_of_items_list = [0] * (number_of_pins + 1)
    
    number_of_deleted_items = 0
    for i in range(len(Y_filtered)) :
        temp_offset = 0;
        i = i - number_of_deleted_items
        for j in range(number_of_pins + 1):
            temp_offset +=offset
            #print("temp_offset = "+ str(temp_offset))
            #print("size of len(Y_filtered) = "+ str(len(Y_filtered))+ "i = "+ str(i))
            if(abs(Y_filtered[i]) <= temp_offset):
                if(number_of_items_list[j] >= number_of_items):
                    del X_filtered[i]
                    del Y_filtered[i]
                    number_of_deleted_items += 1
                else:
                    number_of_items_list[j] +=1
                break
            
    plt.figure()
    plt.hist(Y_filtered, bins=50)
    plt.title("Training dataset Steering command Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(r'./data_histogram_filtered.png')
    plt.show('Training dataset Steering command Histogram.png')
                 
             
    return np.array(X_filtered),np.array(Y_filtered)

    

if __name__ == '__main__':

    dataset_paths = []
    dataset_paths.append(r'./driving_log.csv')
    #train_generator = generator.dataset_generator(paths = dataset_paths, correction = 0.15, batch_size=250)
    
    print("Dataset Preprocessing ....")
    X_train , Y_train = preprocess(dataset_paths)
    
    #X_train = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
    #Y_train = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
    
    #print (X_train)
    #Y_train = range(1,20)
    x_out, y_out =filter(X_train,Y_train , number_of_items=1000)
    
    plt.figure()
    plt.hist(y_out, bins=50)
    plt.title("Training dataset Steering command Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(r'./data_histogram.png')
    plt.show('Training dataset Steering command Histogram.png')

    print("Dataset Preprocessing Done!")
    