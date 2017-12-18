# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_files/model.png "Model Visualization"
[image2]: ./writeup_files/histogram_before_filtering.png "histogram_before_filtering"
[image3]: ./writeup_files/data_histogram_filtered.png "data_histogram_filtered"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create the model
* train.py to train the model.
* dataset_preparator.py This contains the functions that reads the training data, augment it and filter the histogram.
* drive.py for driving the car in autonomous mode
* model_7.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model_7.h5
```

#### 3. Submission code is usable and readable

The model.py ,train.py and dataset_preparator.py files contain the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is adapted from NVidia model with some tunings and removing one convolution layer as show below.

                                                          
![alt text][image1]

The model includes ELU layers to introduce nonlinearity,and the data is normalized in the model using a Keras lambda layer . 

#### 2. Attempts to reduce overfitting in the model

To avoid over fitting I got 2 attempts

* Added dropout layer after each layer.
* Replaced the dropout layers with adding regulizers for each layer and it worked better.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model as mentioned above.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I followed the following strategies 

*  Recording more data at the desired spots.
*  Taking care of the training dataset histogram to be balanced, This will be descriped below.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...
* Cropping2D
* Lambda for normalization                                                                               
* Convolution2D(8, 5, 5, subsample=(2, 2), init='he_normal', activation='elu',W_regularizer=regularizers.l2(0.001))) 
* Convolution2D(20, 5, 5, subsample=(2, 2), init='he_normal', activation='elu',W_regularizer=regularizers.l2(0.001)))
* Convolution2D(35, 5, 5, subsample=(2, 2), init='he_normal', activation='elu',W_regularizer=regularizers.l2(0.001)))
* Convolution2D(64, 3, 3, subsample=(1, 1), init='he_normal', activation='elu',W_regularizer=regularizers.l2(0.001)))
* Flatten                                                                                                                   
* Dense(100, activation='elu', init='he_normal',W_regularizer=regularizers.l2(0.01)))                                                                                                                                          
* Dense(50, activation='elu', init='he_normal',W_regularizer=regularizers.l2(0.001)))                                                                                                             
* Dense(20, init='he_normal',W_regularizer=regularizers.l2(0.001)))                                     
* Dense(1, init='he_normal',W_regularizer=regularizers.l2(0.001)))                                     

#### 3. Creation of the Training Set & Training Process

I captured the training dataset as following 
* 4 laps driving in the middle of the road forward direction.
* 2 laps driving in the middle of the road backward direction.
* 2 laps recovering from the sides.
* Recording more data for the rare situations like the bridge and the road with lane line in one side.

I augmented the data using the following:-
* Flipping the input image and multiplying the steering angles by -1.
* Adding the right and left pictures with correction value of .1

After the collection process, I had the following histogram 

![alt text][image2]
 So I needed to filter the dataset to have balanced histogram, I filtered it by dividing the data set to 500 pins each pin allowed to have only 25 entry.
after filtering the histogram becomes as following

![alt text][image3]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
