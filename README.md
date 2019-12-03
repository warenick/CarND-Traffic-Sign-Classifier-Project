# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/classesDisturb.png "Visualization"
[image2]: ./images/datasetImgs.png "Datase"
[image3]: ./images/nomalize.png "normalize"
[image4]: ./images/contrast.png "contrast"
[image5]: ./traffic-signs-data/not_in_dataset/2.bmp "Traffic Sign 1"
[image6]: ./traffic-signs-data/not_in_dataset/3.bmp "Traffic Sign 2"
[image7]: ./traffic-signs-data/not_in_dataset/6.bmp "Traffic Sign 3"
[image8]: ./traffic-signs-data/not_in_dataset/8.bmp "Traffic Sign 4"
[image9]: ./traffic-signs-data/not_in_dataset/10.bmp "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/warenick/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First of all I looked at the dataset. Here is an example of images:

![alt text][image2]

As a first step, I normalize the images because some images are very dark and some are overexposed

Here is an example of a traffic sign image before and after nomalize.

![alt text][image3]

As a last step, I apply clahe filter to increase contrast of the images because many images have low contrast

Here is an example of a traffic sign image before and after increasing contrast.

![alt text][image4]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I start with classic LeNet model architecture.

LeNet(
  
  (conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
  
  (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
  
  (fc1): Linear(in_features=576, out_features=120, bias=True)
  
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  
  (fc3): Linear(in_features=84, out_features=43, bias=True)

)

My final model consisted of the following layers:

Net2(

  (layer1): Sequential(
  
    (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  
    (1): ReLU()
  
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )

  (layer2): Sequential(
  
    (0): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  
    (1): ReLU()
  
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )

  (drop_out): Dropout(p=0.5, inplace=False)
  
  (fc1): Linear(in_features=4096, out_features=1000, bias=True)
  
  (fc2): Linear(in_features=1000, out_features=420, bias=True)
  
  (fc3): Linear(in_features=420, out_features=43, bias=True)
)
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an SGD optimizer, batch size of 128 shuffled elements and 25 epochs. My hyperparameters are learning rate=0.005 and momentum=0.9. 
For training i use my gpu device.

criterion = CrossEntropyLoss()

Epoch: 1/25

Loss: 113.611 

Accuracy : 42 %

Epoch: 2/25

Loss: 70.385 

Accuracy : 49 %

Epoch: 3/25

Loss: 56.375 

Accuracy : 57 %

Epoch: 4/25

Loss: 49.625 

Accuracy : 56 %

Epoch: 5/25

Loss: 42.056 

Accuracy : 66 %

Epoch: 6/25

Loss: 37.833 

Accuracy : 71 %

Epoch: 7/25

Loss: 35.830 

Accuracy : 76 %

Epoch: 8/25

Loss: 28.586 

Accuracy : 79 %

Epoch: 9/25

Loss: 28.238 

Accuracy : 68 %

Epoch: 10/25

Loss: 24.153 

Accuracy : 81 %

Epoch: 11/25

Loss: 23.252 

Accuracy : 85 %

Epoch: 12/25

Loss: 22.290 

Accuracy : 82 %

Epoch: 13/25

Loss: 18.441 

Accuracy : 87 %

Epoch: 14/25

Loss: 16.664 

Accuracy : 89 %

Epoch: 15/25

Loss: 15.185 

Accuracy : 73 %

Epoch: 16/25

Loss: 14.019 

Accuracy : 76 %

Epoch: 17/25

Loss: 12.780 

Accuracy : 91 %

Epoch: 18/25

Loss: 11.406 

Accuracy : 88 %

Epoch: 19/25

Loss: 10.253 

Accuracy : 91 %

Epoch: 20/25

Loss: 9.180 

Accuracy : 88 %

Epoch: 21/25

Loss: 9.152 

Accuracy : 94 %

Epoch: 22/25

Loss: 6.881 

Accuracy : 93 %

Epoch: 23/25

Loss: 7.234 

Accuracy : 96 %

Epoch: 24/25

Loss: 5.905 

Accuracy : 90 %

Epoch: 25/25

Loss: 5.384 

Accuracy : 95 %

finifh training

Accuracy of the network validation images: 93 %

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 97%
* validation set accuracy of 93%
* test set accuracy of 95%

<!-- TODO: -->
If an iterative approach was chosen:
the first architecture I chose was classic LeNet. It is well described and easy to implement, but this network is too small for high-quality sign processing.
I increased the number of links in the network and the size of the convolutional layer window to improve the definition of Fitch in the image.
This allowed us to achieve a better result.
I tuned batch_size to increase the learning speed and transferred the learning process to the video card, which allowed to speed up the process a lot
When training I heuristically changed the learning rate and the number of epoch to achieve a better result without retraining. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

The second image might be difficult to classify because they overexposed, and last image might be difficult to classify because they have a small twist.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 80 km/h	| Speed limit 80 km/h							| 
| Stop       			| Stop   										|
| 3.5 tonns prohibited	| 3.5 tonns prohibited							|
| Main road	      		| Yield     					 				|
| Go straight or right	| Go straight or right 							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a Speed limit 80 km/h sign (probability of 0.6), and the image does contain a stop sign. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .20         			| Speed limit 80 km/h   									| 
| .17     				| Speed limit 50 km/h 										|
| .16					| Speed limit 100 km/h											|

For the other images:
Probabilitys of types

probabilitys
[20.6604, 17.7442, 16.4139]
of
[ 5,  2,  7]
5 - is true

probabilitys
[13.1171,  8.3251,  8.2044]
of
[14, 15, 26]
14 - is true

probabilitys
[22.0055, 17.7831, 17.0745]
of
[16, 12, 41]
16 - is true

probabilitys
[11.8150, 11.4877,  7.6204]
of
[13, 12,  5]
12 - is true

probabilitys
[26.5168, 18.5674, 18.3341]
of
[36, 17, 13]
36 - is true


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


