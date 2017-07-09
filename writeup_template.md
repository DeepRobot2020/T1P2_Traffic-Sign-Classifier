#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image_train]: ./write_up_images/valid_data_bar.png "train Data"
[image_valid]: ./write_up_images/valid_data_bar.png "Valid Data"
[image_test]: ./write_up_images/valid_data_bar.png "test Data"

[img1]: ./new_images/speed_limit_70.jpg "Traffic Sign 1"
[img2]: ./new_images/end_of_speed_limit.jpg "Traffic Sign 2"
[img3]: ./new_images/speed_limit_100.jpg "Traffic Sign 3"
[img4]: ./new_images/double_curve.jpg "Traffic Sign 4"
[img5]: ./new_images/stop.png "Traffic Sign 5"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/darknight1900/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the label distribution inside train, valid, test dataset.

Training data Set:
![alt text][image_train]
Validation data Set:
![alt text][image_valid]
Test dataset
![alt text][image_test]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to not convert the images to grayscale because grayscale image produced worse performance with my modified LeNet network. 

The only preprocessing I did is to normalize the image into range of [-1.0, 1.0]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|										
| Max pooling 2x2	    | 2x2 stride,  outputs 14x14x12 |				
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x12 	
| RELU  |     
| Max pooling 2x2	    | 2x2 stride,  valid padding,outputs 5x5x12 |
| Flattern	    | outputs 300 |
| Fully connected	    | outputs 240 |
| Fully connected	    | outputs 84 |
| Fully connected	    | outputs 43 |
      |

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer, batch size 128, number of epoches 30 and learning rate 0.001

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997?
* validation set accuracy of 0.947 
* test set accuracy of 0.935



If a well known architecture was chosen:
* What architecture was chosen? (LeNet, with more depth on first layer CNN)
* Why did you believe it would be relevant to the traffic sign application? (The problems are similar: we are trying to classify images. However, the traffic sign problems have more details, that's why I add more depth in the first CNN layer )

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well (With epoches 30, I am able to achieve 0.94 prediction accuracy )?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][img1] ![alt text][img2] ![alt text][img3] 
![alt text][img4] ![alt text][img5]

* For image 1, it might be diffculty as there are some trees in the background and that could make the prediction hard. 

* For image 2, it might be diffculty as there are some trees in the background and that could make the prediction hard. 

* For image 3, it might be diffculty  

* For image 4, it might be diffculty as there are some trees in the background and that could make the prediction hard. 

* For image 5, it might be diffculty as there are some trees in the background and that could make the prediction hard. 


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 70      		| Speed limit 70   									| 
| End of all speed limit    			| End of no passing 										|
| Speed limit 100					| Speed limit 100											|
| Stop Sign	      		| Stop Sign					 				|
| Double Curve			| Dangerous cure to right      							|

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This result is wrose, compared to the prediction accuray on test dataset (93%). However, if the number of test images increase, the prediction accuray should be close to 93%. The reviewer also kindly pointed out that this might be caused by unblanced training data set and the issue can be resolved by adding jitter into traning dataset. I will try this trick later on. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a stop sign (probability of 0.99), and the image does contain a stop sign. The top five soft max probabilities were

![alt text][img1]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9956         			| Speed Limit 70   									| 
| .0040     				| General Caution 										|
| .0					| Keep left											|
| .0	      			| Go straight or left					 				|
| .0				    | Wild animals crossing |    							|
|

For the second image, the correct prediction should be 'End of speed limit (80km/h)', but the network failed to predict 
the correct results. 

![alt text][img2]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .56         			| End of no passing   									| 
| .0044     				| End of all speed and passing limits 										|
| .0					| End of speed limit (80km/h)											|
| .0	      			| Spped limit					 				|
| .0				    | End of no passing by vehicles over 3.5 metric tons |    							|
|

For the third image, the model very sure it is a 'Speed limit (100km/h)' and this prediction is correct.

![alt text][img3]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (100km/h)   									| 
| 0.0     				| Speed limit (120km/h) 										|
| .0					| Speed limit (50km/h)										|
| .0	      			| Speed limit (30km/h)					 				|
| .0				    | Vehicles over 3.5 metric tons prohibited |    							|
|


For the fouth image, the model failed to predict that this is a 'Dangerous curve to the right'. This is in fact a very intersting result. Comparing the correct result and predict result, two images are very similar. The original image size was 255x255 and when I feed the image into my network, I resized the image to 32x32 and during this process, many details in the original image has lost. I believe this is the major reason why the prediction result is wrong. 

![alt text][img4]


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9945         			| Dangerous curve to the right   									| 
| 0.0047     				| Children crossing 										|
| .00007					| Beware of ice/snow										|
| .0	      			| Right-of-way at the next intersection					 				|
| .0				    | Slippery road |    							|
|

For the last image, the prediction from my model is prery good. It guesses that the input image is a 'Stop' sign with 0.9945 probability. 

![alt text][img5]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9945         			| Stop   									| 
| 0.0047     				| Bicycles crossing 										|
| .00007					| Speed limit (30km/h)										|
| .0	      			| Priority road intersection					 				|
| .0				    | Speed limit (80km/h) |    							|
|
