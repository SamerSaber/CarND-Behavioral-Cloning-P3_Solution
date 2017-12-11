    #import data_generator
import model
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
import dataset_preparator
import generator
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

print("Creating the model.")
# Create Instance of Custom Nvidia Model
model = model.nvidia_model((160, 320, 3),((50,20), (0,0)))

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
#dataset_paths.append('./training_data/Forward_track/driving_log.csv')
#dataset_paths.append('./training_data/Backward_track/driving_log.csv')
#dataset_paths.append('./training_data/Sides_recovery/driving_log.csv')
dataset_paths.append(r'./driving_log.csv')
train_generator = generator.dataset_generator(paths = dataset_paths, correction = 0.15, batch_size=250)

#print("Dataset Preprocessing ....")
#X_train , Y_train = dataset_preparator.preprocess(dataset_paths)
#print("Dataset Preprocessing Done!")


#history_object = model.fit(X_train, Y_train, validation_split=.2, shuffle=True, nb_epoch=8 , batch_size=128)


history_object = model.fit_generator(
                                        train_generator,
                                        samples_per_epoch=27750,
                                        nb_epoch=8,
                                        #validation_data=valid_generator,
                                        nb_val_samples=1664  # , callbacks=[earlystopping_cb]

                                    )

model.save(r'model_7.h5')

# plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
# plt.savefig('foo.png')