from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation, MaxPooling2D, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras import regularizers

def nvidia_model(input_shape , crop_shape = ((0,0),(0,0))):

    model = Sequential()
    model.add(Cropping2D(cropping = crop_shape, input_shape = input_shape))
    model.add(Lambda(lambda x: (x / 255.0 - 0.5)))
    model.add(Convolution2D(8, 5, 5, subsample=(2, 2), init='he_normal', activation='elu',W_regularizer=regularizers.l2(0.001))) 
    model.add(Convolution2D(20, 5, 5, subsample=(2, 2), init='he_normal', activation='elu',W_regularizer=regularizers.l2(0.001)))
    model.add(Convolution2D(35, 5, 5, subsample=(2, 2), init='he_normal', activation='elu',W_regularizer=regularizers.l2(0.001)))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init='he_normal', activation='elu',W_regularizer=regularizers.l2(0.001)))
    
   
    model.add(Flatten())
    #model.add(Dense(100, activation='elu', init='he_normal',W_regularizer=regularizers.l2(0.01)))


    model.add(Dense(50, activation='elu', init='he_normal',W_regularizer=regularizers.l2(0.001)))
#     model.add(Dropout(0.5))
    
    model.add(Dense(20, init='he_normal',W_regularizer=regularizers.l2(0.001)))
 
    
    model.add(Dense(1, init='he_normal',W_regularizer=regularizers.l2(0.001)))
    #model.add(Dropout(0.5))
    return model