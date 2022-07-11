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

CODE

Make a train.py file and include the following code in it:

***************************************************************
'''
import numpy as np
import cv2
#from keras.emotion_models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

#train_dir = 'data/train'
train_dir = 'D:/Dev/py_proj/emojify/archive/train'
#val_dir = 'data/test'
val_dir = 'D:/Dev/py_proj/emojify/archive/test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=64,
        #color_mode="gray_framescale",
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

#CNN architecture
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
# emotion_model.load_weights('emotion_model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

#Training
emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
emotion_model_info = emotion_model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=7178 // 64)

#saving weights and trained model in emotion_model.h5 file
emotion_model.save_weights('emotion_model.h5')

# start the webcam feed
cap = cv2.VideoCapture(0)
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    bounding_box = cv2.CascadeClassifier('D:/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
****************************************************************

Now make a gui.py file and add the following code:

*****************************************************************
'''
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np

import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))#Activation relu means that the activation function is ReLU(Rectified Linear Unit)
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('emotion_model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}


#emoji_dist={0:"./emojis/angry.png",2:"./emojis/disgusted.png",2:"./emojis/fearful.png",3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surpriced.png"}
#emoji_dist={0:"./emojis/angry.png",1:"./emojis/disgusted.png",2:"./emojis/fearful.png",3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surpriced.png"}
emoji_dist={0:"D:/Dev/py_proj/emojify/emoji-creator-project-code/emojis/angry.png",1:"D:/Dev/py_proj/emojify/emoji-creator-project-code/emojis/disgusted.png",2:"D:/Dev/py_proj/emojify/emoji-creator-project-code/emojis/fearful.png",3:"D:/Dev/py_proj/emojify/emoji-creator-project-code/emojis/happy.png",4:"D:/Dev/py_proj/emojify/emoji-creator-project-code/emojis/neutral.png",5:"D:/Dev/py_proj/emojify/emoji-creator-project-code/emojis/sad.png",6:"D:/Dev/py_proj/emojify/emoji-creator-project-code/emojis/surpriced.png"}
global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text=[0]

def show_vid():      
    cap1 = cv2.VideoCapture(0)                                 
    if not cap1.isOpened():                             
        print("cant open the camera1")
    flag1, frame1 = cap1.read()
    frame1 = cv2.resize(frame1,(560,500))
    #frame1 = cv2.resize(frame1,(600,500))

    #bounding_box = cv2.CascadeClassifier('/home/shivam/.local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    bounding_box = cv2.CascadeClassifier('D:/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        
        maxindex = int(np.argmax(prediction))
        # cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0]=maxindex
    if flag1 is None:
        print ("Major error!")
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)     
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_vid)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()


def show_vid2():
    frame2=cv2.imread(emoji_dist[show_text[0]])
    pic2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2=Image.fromarray(frame2)
    imgtk2=ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2=imgtk2
    lmain3.configure(text=emotion_dict[show_text[0]],font=('arial',40,'bold'))
    
    lmain2.configure(image=imgtk2)
    lmain2.after(10, show_vid2)

if __name__ == '__main__':
    root=tk.Tk()
    
    img = ImageTk.PhotoImage(Image.open("logo.png"))
    heading = Label(root,image=img,bg='black')
 
    heading.pack() 
    heading2=Label(root,text="E\nM\nO\nJ\nI\nF\nY\nI\nN\nG",pady=20, font=('Comic Sans MS',25,'bold'),bg='black',fg='#CDCDCD')                                 
    
    heading2.pack()
    lmain = tk.Label(master=root,padx=50,bd=10)
    lmain2 = tk.Label(master=root,bd=10)

    lmain3=tk.Label(master=root,bd=10,fg="#CDCDCD",bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=50,y=250)
    lmain3.pack()
    lmain3.place(x=960,y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900,y=350)
    


    root.title("Image to emoji")            
    root.geometry("1400x900+100+10") 
    root['bg']='black'
    exitbutton = Button(root, text='Quit',fg="red",command=root.destroy,font=('arial',25,'bold')).pack(side = BOTTOM)
    
    #show_vid()
    #show_vid2()
    #root.mainloop()
show_vid()
show_vid2()
root.mainloop()
'''
***************************************************************
