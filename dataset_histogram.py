import numpy as np
import csv
import matplotlib.pyplot as plt

dataset_paths = []
dataset_paths.append('./training_data/Forward_track/driving_log.csv')
dataset_paths.append('./training_data/Backward_track/driving_log.csv')
dataset_paths.append('./training_data/Sides_recovery/driving_log.csv')


training_steering = []
for path in dataset_paths :
    with open(path) as log_csv:
        log_reader = csv.reader(log_csv)
        for line in log_reader:
            training_steering.append(float(line[3]))



training_dataset_size = len(training_steering)

print('Training Dataset size =', training_dataset_size)

plt.figure()
plt.hist(training_steering, bins=100)
plt.title("Training dataset Steering command Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")


plt.show()