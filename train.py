    #import data_generator
import model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import dataset_preparator
import generator
import numpy as np
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import keras 

print("Creating the model.")
# Create Instance of Custom Nvidia Model
model = model.nvidia_model((160, 320, 3),((60,20), (0,0)))

print("Creating the optimizer.")
# define training optimizer
adam = Adam(lr=5e-4)
optimizer = adam

print("Compiling the model.")
# Compiling model
model.compile(optimizer, loss="mse")

# Print Model Summary.
model.summary()
dataset_paths = []

dataset_paths.append(r'./training_data/Forward_track/driving_log.csv')
dataset_paths.append(r'./training_data/Backward_track/driving_log.csv')
dataset_paths.append(r'./training_data/Sides_recovery/driving_log.csv')
#dataset_paths.append(r'./driving_log.csv')
rare_situation_paths = []
dataset_paths.append(r'./training_data/Rare_Situations/driving_log.csv')

print("Dataset Preprocessing ....")
X_train , Y_train = dataset_preparator.preprocess(dataset_paths)
x_out, y_out =dataset_preparator.filter(X_train,Y_train , number_of_items=40 , number_of_pins= 501)
#x_out, y_out =dataset_preparator.filter(x_out,y_out , number_of_items=25 , number_of_pins= 501)
print("Dataset Preprocessing Done!")
#X_rare, Y_rare = dataset_preparator.preprocess(rare_situation_paths)
 
#x_out = np.concatenate((x_out,X_rare))
#y_out = np.concatenate((y_out,Y_rare))
plt.figure()
plt.hist(y_out, bins=501)
plt.title("Training dataset Steering command Histogram Final")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.savefig(r'./data_histogram_final.png')
plt.show('Training dataset Steering command Histogram Final.png')

ES = keras.callbacks.EarlyStopping(monitor = 'val_loss' , min_delta = 0.0001 ,patience = 3, verbose = 0, mode = 'auto')
callbacks_list = [ES]
history_object = model.fit(x_out, y_out, validation_split=.2, shuffle=True, nb_epoch=100 , batch_size=164, callbacks = callbacks_list)



model.save(r'model_10.h5')


#plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
# plt.savefig('foo.png')
plt.savefig('./loss.png')
plt.show()