# CNN-Implementation-Using-Python
Implementation of an ML algorithm - CNN(Convolutional Neural Network) using Python.

WHAT IS CNN?

It is a deep learning algrithm which is commonly used for image processing and recognition.
It proves efficient in dealing with large datasets and its analysis.

![CNN_neuralnetwork](https://user-images.githubusercontent.com/84243839/178183853-480d76bf-26eb-4a26-98f3-fa7c1c4fb1d9.png)

The CNN model is inspired from the structure of the brain. The many millions of single cell neurons
constitute and form the brain.

![Structure-Neuron](https://user-images.githubusercontent.com/84243839/178189472-059eaa46-b283-4139-8e06-e6ce41fa354b.jpg)


Similarly the smallest single unit of an artificial neural network is called the perceptron and is
used to form single layer and multilayered neural networks.

It is important to brief about the various parts of the perceptron before moving ahead.

The perceptron itself, consists of weights, the summation processor, and an activation function, and 
an adjustable threshold processor called bias.

![perceptron](https://user-images.githubusercontent.com/84243839/178183839-d2bacbb2-644c-445a-a096-9e3ae7aec85a.png)

-> Here the bias is the input 1, it can be thought of as a tendency towards a particular way of behaving.
   It can also be said that the bias is the internal systematic error of the neural network caused by itself
   in calculating the output.

-> The activation function is the non linear transformation. It decides whether the perceptron should fire or not.
   Sigmoid function and Step function are examples of activation functions.  

In this way the output generated is passed on to the next perceptron as input and so on.


Moving ahead,
In mathematics (in particular, functional analysis), convolution is a mathematical operation on 
two functions(say f and g) that produces a third function(f*g) that expresses how the shape of one
is modified by the other. The term convolution refers to both the result function and to the process
of computing it.

Images are nothing but matrix pixel values. CNN can work with both RGB and grayscale images. RGB have
3 planes of matrix pixel values (1 plane for each color) whereas grayscale has only 1 plane, this makes
working with grayscale images easier.


![convolution](https://user-images.githubusercontent.com/84243839/178197934-2aae1339-b329-46e2-b38a-c2ed383bb9a7.png)


Technically convolution is done on these pixel matrices and the result is passed on to the other layers
of the neural network.

# About This Project
In this project the CNN model using python will be built that recognizes facial emotions and generates an 
avatar associated to that emotion. The model is trained on a dataset of images to classify facial expressions 
and mapping of the expression to an avatar/emoji is done.

![Screenshot (130)](https://user-images.githubusercontent.com/84243839/178207888-11eba9b2-91f6-4eb5-b7b0-152584ee21f5.png)

DATASET

The model works on FER2013 which is a dataset of 48 * 48 pixel grayscale face images. This dataset consists of
7 emotion categories - angry, disgust, fear, happy, sad, surprise, neutral.

METHOD

There will be 2 files, train.py to train the model and gui.py file to create an interface for the user.

Important libraries that are used are:

Numpy library,

Keras: It is a deep learning API written in Python, running on top of the machine learning platform TensorFlow.

OpenCV: It is an image processing library, where AI algorithms can be used for image recognition and processing.

Tkinter: To build GUI.

