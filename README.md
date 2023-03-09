# Visual-Emotion-Detection
Emotion Detection using Transfer Learning Models

# Live Class Monitoring System(Facical Expression Recognition)

## Introduction
Facial emotion recognition is the process of detecting human emotions from facial expressions. The human brain recognizes emotions automatically, and software has now been developed that can recognize emotions as well. This technology is becoming more accurate all the time, and will eventually be able to read emotions as well as our brains do. 

AI can detect emotions by learning what each facial expression means and applying that knowledge to the new information presented to it. Emotional artificial intelligence, or emotion AI, is a technology that is capable of reading, imitating, interpreting, and responding to human facial expressions and emotions.

Facial expressions are a form of nonverbal communication. Various studies have been done for the classification of these facial expressions. There is strong evidence for the universal facial expressions of seven emotions which include: neutral happy, sadness, anger, disgust, fear, and surprise. So it is very important to detect these emotions on the face as it has wide applications in the field of Computer Vision and Artificial Intelligence. These fields are researching on the facial emotions to get the sentiments of the humans automatically.
## Problem Statement

Safety monitoring: Emotional recognition models can be used to detect signs of stress or distress in individuals working in high-risk environments, such as emergency responders or military personnel. This can help prevent accidents and improve overall safety.

According to a study published in the Journal of Occupational Health Psychology, workplace stress is a significant predictor of safety incidents, accidents, and near-misses. The study found that workers who reported high levels of stress were more likely to experience safety incidents and that this relationship was particularly strong for individuals working in high-risk industries such as emergency response and military operations. This highlights the importance of monitoring and managing workplace stress to improve safety outcomes. Emotional recognition models have shown promise in detecting signs of stress and distress in individuals. For example, a recent study published in the journal Sensors found that an emotional recognition model based on facial expressions was able to accurately detect stress in emergency responders with an accuracy of 87%.

By using emotional recognition models for safety monitoring, organizations can potentially reduce the incidence of safety incidents and improve overall safety outcomes for workers.
While digital platforms have limitations in terms of physical surveillance but it comes with the power of data and machines which can work for you. It provides data in the form of video, audio, and texts which can be analyzed using deep learning algorithms.

Certainly! Workplace safety is a critical concern across a wide range of industries, particularly those involving high-risk environments such as emergency response, military operations, or hazardous workplaces. These environments can be physically demanding, stressful, and mentally taxing, which can contribute to safety incidents and accidents. Research has shown that workplace stress is a significant predictor of safety incidents and that managing workplace stress is essential for improving safety outcomes.

I will solve the above-mentioned challenge by applying deep learning algorithms to live video data.
The solution to this problem is by recognizing facial emotions.

## Dataset Information

I have built a deep learning model which detects the real time emotions of students workers through a webcam so that a manager can understand if workers are able to cope-up with the work without any stress or some stress' expressions or emotions and then deploy the model. The model is trained on the FER-2013 dataset .This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.
Here is the dataset link:-https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
## Model Creation
# 1)
# Face-Emotion-Recognition Using Transfer Learning
 Here I have created the models which recognizes the real time emotion of person in frame. The project is done by me with using some online help. Here is model trained by me using Transfer Learning with a pre trained model MobileNet.

Transfer learning is a  research problem in machine learning model that focuses on storing knowledge gained while solving a problem and applies it to another problem of similar kind. It offers better starting point and improves the model performance when applied on second task. The model file for this is provided in my_model folder you can have a look.

![transfer_learning](https://user-images.githubusercontent.com/81186352/117619020-613f2c80-b18c-11eb-845a-7396b80aa5ff.jpg)


## You can take reference of my jupyter notebook 
https://github.com/Vagueken/Visual-Emotion-Detection/blob/main/Python%20Files/Tranfer%20Learning%201%20Mobilenet.ipynb
...
 
 In this Model 'MobileNet' Transfer-Learning is used, along with computer vision for Real time face emotion recognition through webcam, so based on these a streamlit app is created which is deployed on Heroku cloud platform and streamlit's own streamllit share platform.
The model is trained on the dataset 'FER-13 cleaned dataset', which had seven emotion categories namely 'Happy', 'Sad', 'Neutral','Angry','Surprise','Fear' and 'Disgust' in which all the images were 48x48 pixel grayscale images of face. This model gave an accuracy of approximately 78% on train data, and around 76% of accuracy on test data at 30th epoc.


 I also trained a model using CNN which gave an accuracy of 66.47% for train data, and 58.19% on test data.


# Dependencies
* Tensorlow
* Keras
* MobileNet
* Opencv
* Streamlit


# Setup
## You need  the Following:
Python and the following packages:
* OpenCV 
* Keras
* Tensorflow
* Numpy
* pandas
* Matplotlib
* sklearn

# 2)
# Emotion-Recognition Web Application With Streamlit(Using Keras and CNN) 
A CNN based Tensorflow implementation on facial expression recognition (FER2013 dataset), achieving 66.47% accuracy 
![](images/model.png) 


## You can take a reference of my jupyter notebook for building model. Here is the link : 
...

### Dependencies:
- python 3.10<br/>
- Keras with TensorFlow as backend<br/>
- Streamlit framework for web implementation

### FER2013 Dataset:
- Dataset from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data<br/>
- Image Properties: 48 x 48 pixels (2304 bytes)<br/>
- Labels: 
> * 0 - Angry :angry:</br>
> * 1 - Disgust :anguished:<br/>
> * 2 - Fear :fearful:<br/>
> * 3 - Happy :smiley:<br/>
> * 4 - Sad :disappointed:<br/>
> * 5 - Surprise :open_mouth:<br/>
> * 6 - Neutral :neutral_face:<br/>
- The training set consists of 28,708 examples.<br/>
- The model is represented as a json file :model.json
The separated dataset is already available to download in the two folders train and test.

loss_accuracy_plot.png![image](https://user-images.githubusercontent.com/67421545/224051947-1a344749-ef09-4c88-98e0-cd3bb449feb1.png)

# This was the CNN model that gave low slug size,both of them are present in my github repository.

## Realtime Local Video Face Detection

I created  patterns for detecting and predicting single faces and as well as multiple faces using OpenCV videocapture in local.
For Webapp , OpenCV can’t be used. Thus, using Streamlit for front-end application.Also because streamlit was taking a lot of time to boot so we also deployed it in flask just so that we have a working model that is quickly deployed.


## Deployment of Streamlit WebApp in Streamlit

In this repository I have made a front end using streamlit as it have recently launched a streamlit.share platform.


Streamlit Link:- 
https://vagueken-real-time-face-emotion-detection-streamlit-app-2vopec.streamlit.app/


## Conclusion

Finally I build the webapp and deployed which has training accuracy of 78% and test accuracy of 76% .



