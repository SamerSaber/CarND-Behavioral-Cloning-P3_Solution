#import data_generator
import nvidia_model
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
import dataset_preparator

print("Creating the model.")
# Create Instance of Custom Nvidia Model
model = nvidia_model.nvidia_model((160, 320, 3),((70,20), (0,0)))

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
dataset_paths.append('./training_data/Forward_track/driving_log.csv')
dataset_paths.append('./training_data/Backward_track/driving_log.csv')
dataset_paths.append('./training_data/Sides_recovery/driving_log.csv')


print("Dataset Preprocessing ....")
X_train , Y_train = dataset_preparator.preprocess(dataset_paths)
print("Dataset Preprocessing Done!")
history_object = model.fit(X_train, Y_train, validation_split=.2, shuffle=True, nb_epoch=8 , batch_size=64)
model.save('model_5.h5')

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
