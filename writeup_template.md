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

[image1]: ./examples/center.jpg "center driving"
[image2]: ./examples/left_side1.jpg "left recovery 1"
[image3]: ./examples/left_side2.jpg "left recovery 2"
[image4]: ./examples/left_side3.jpg "left recovery 3"
[image5]: ./examples/left_side4.jpg "left recovery 4"
[image6]: ./examples/left_side5.jpg "left recovery 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

-> i explicitly did not use a python generator due to heavy performance issues. i really liked the idea of preparing a preprocessing pipeling outside of the network and make it reusable through a generator and meanwhile saving memory. but unfortunately training with such an generator took literally forever. i don't know yet who or why but will investigate this for future applications. just to let you know.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I tried a few different network layouts, but then i decided to try the one from NVIDIA mentioned during the lessons, which gave me a pretty good starting point. From there i tried some minor modifications to that layout, such as adding dropouts or MaxPooling, as well as changing the architecture. 
In the end the modification i added is  an 'oscillation' of MaxPooling and dropouts between the convolutional layers. When i added a MaxPooling layer, i changed the strides of the convolutional layers to 1x1, whereas i set it to 2x2 when a dropout was following, in order to make the output dimensions fit into the following layers.

My model consists of a convolution neural network with 3 5x5 filters followed by 2 3x3 filters and depths between 24 and 64 (model.py lines 69-77). The model includes RELU layers to introduce nonlinearity (69-83) and MaxPooling or dropouts between the convolutions. Also the data is normalized in the model using a Keras lambda layer (code line 67) as well as cropped (to focus on a region of interest within the images) (code line 68)

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| keras lambda   		| normalization  (x / 255.0) - 0.5			| 
| cropping       		| cut out region of interest (dropped 70px from top and 25 px from bottom)	| 
| Convolution 5x5    		| depth 24, 1x1 stride, same padding 	| 
| RELU					|												|
| Max pooling	      	| 2x2 pool_size, 2x2 stride		|
| Convolution 5x5    		| depth 36, 2x2 stride, same padding 	| 
| RELU					|												|
| Convolution 5x5    		| depth 48, 1x1 stride, same padding 	| 
| RELU					|												|
| Max pooling	      	| 2x2 pool_size, 2x2 stride		|
| Convolution 3x3    		| depth 64, 2x2 stride, same padding 	| 
| RELU					|												|
| Convolution 3x3    		| depth 64, 1x1 stride, same padding 	| 
| RELU					|												|
| Max pooling	      	| 2x2 pool_size, 2x2 stride		|
| Fully connected		| Output 1164 |
| RELU					|												|
| Fully connected		| Output 100 |
| RELU					|												|
| Fully connected		| Output 50 |
| RELU					|												|
| Fully connected		| Output 10 |
| RELU					|												|
| Fully connected		| --> Final Output 1 |



#### 2. Attempts to reduce overfitting in the model

The model contains dropouts (model.py lines 72 and 76) and MaxPooling (model.py lines 70, 74 and 78) layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 87). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 86).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road as well as driving the track in opposite directions. I also repeated driving critical parts of the road. I also artificially augmented the training data, by flipping the given images and inverting the steering angles accordingly as well as using the images from the left and right camera with some fixed offset for the steering angle. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to first try some simple small architechtures, which didn't go very well overall. as stated before, i later tried some well known architechtures like the one from NVIDIA, which became the final starting point for fine tuning in this project.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, like the first left corner. there the car went straight into the see. next hurdle was the bridge, due to the different road texture. obviously most of the road had a similay texture, wherea the bridge differed. due to the natural inbalance of training data when just driving simple circles, the bridge was problematic. so i recorded more training data, crossing the bridge only.
next hot point became the corner after the bridge, where the edge of the road was not clearly marked, since there was a small way out onto some sandy road. in this case it also helped to record a few more attemts of taking the corner accordingly, as well as introducing 'hard recoveries' like facing directly to the offtrack start recording and then turn hard to the left, back to the road.
last but not least, the double curve close to the inner see was hard to get for quite a while. but in this case it was not necessary to rerecord something. instead i tweaked the network architecture itself, as well as added more steps to the data augmentation. e.g. using left and right camera images combined with some manually given offset to the steering data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture listed in the table above. (sorry for not having a visualization of the architechture, but i did not find an appropriate tool to draw such a diagram.. maybe that would be a good point for the course, since i would like to have a design scheme of the architecture)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. i did this in both directions. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover in case it looses track over time. that happens e.g. if you only collect training data with super small steering angles, since the model learns that big steering angles are never used! So in fact too good driving in the simulator might lead to some kind of overfitting. These images show what a recovery looks like starting from the left of the road back to the center:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]


As mentioned above, i added recordings of specific situations, where problems in autonomous mode occured. 

To augment the data sat, I also flipped images and angles thinking that this would additionally avoid overfitting. I also mentioned most of that in the former section.


After the collection process and additional augmentation, I had 19649 data points. I then preprocessed this data by first changing the color space to RGB and then (within the model) normalizing and cropping it.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.
