import tkinter as tk
from tkinter import filedialog
from tkinter import *
from sklearn import metrics
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

# donwload haarcascade_frontalface_default from here "https://github.com/opencv/opencv/tree/master/data/haarcascades"

def FacialExpressionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
    return model

top =tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model_a.json","model.weights.h5")

EMOTIONS_LIST = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

def Detect(file_path):
    global Label_packed
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image,1.3,5)
    try:
        for (x,y,w,h) in faces:
            fc = gray_image[y:y+h,x:x+w]
            roi = cv2.resize(fc,(48,48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]
        print("Predicted Emotion is " + pred)
        label1.configure(foreground="#011638",text = pred)
    except:
        label1.configure(foreground="#011638",text = "Unable to detect")

def show_Detect_button(file_path):
    detect_b = Button(top,text="Detect Emotion", command= lambda: Detect(file_path),padx=10,pady=5)
    detect_b.configure(background="#563641",foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx =0.79,rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except:
        pass

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#415636",foreground='white',font=('arial',20,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top,text='Emotion Detector',pady=20,font=('arial',25,'bold'))
heading.configure(background='#CDCDCD',foreground="#563651")
heading.pack()
top.mainloop()

"""
tkinter: This is Python’s standard GUI (Graphical User Interface) package.
filedialog: A module within tkinter to allow users to select files via a dialog box.
sklearn.metrics: Contains various machine learning metrics (though in this code, it doesn't seem to be used, so it might be unnecessary).
model_from_json: A function from TensorFlow that allows loading a model’s architecture from a JSON file.
PIL.Image and ImageTk: Used for opening, manipulating, and displaying images in the GUI.
numpy: A powerful numerical computing library.
cv2: The OpenCV library for computer vision tasks like face detection.

Purpose: This function loads a pre-trained deep learning model that is used to detect emotions from faces.
Steps:
Loading the Model Architecture: The model structure is read from a JSON file.
Loading the Weights: The trained weights are loaded into the model.
Compiling the Model: The model is compiled with an optimizer (adam), a loss function (categorical_crossentropy), and an evaluation metric (accuracy).
Return: The function returns the compiled model ready for prediction.

Purpose: This section sets up the main window for the GUI.
Details:
Tk(): Initializes the main window.
geometry(): Sets the size of the window.
title(): Sets the window title.
configure(): Changes the background color.

label1: A label for displaying the predicted emotion.
sign_image: A label where the uploaded image will be shown

Purpose: This loads the pre-trained Haar Cascade Classifier for face detection from an XML file. It’s a machine learning-based approach to detect objects in images, here specifically for faces.
Purpose: The model is loaded using the previously defined FacialExpressionModel function. This model will be used to predict emotions based on facial images.

Steps:
Read Image: The image file selected by the user is read.
Convert to Grayscale: The image is converted to grayscale because face detection often works better on grayscale images.
Face Detection: The detectMultiScale method detects faces in the image.
Region of Interest (ROI): For each detected face, a region of interest (the face) is extracted.
Resize: The face region is resized to the input shape expected by the model (48x48 pixels).
Normalization: The pixel values are normalized by dividing by 255 to bring them into the range [0, 1].
Expand Dimensions: The array is reshaped to match the input dimensions required by the model.
Prediction: The model predicts the emotion, and the result is displayed in the GUI.
Error Handling: If no face is detected, or an error occurs, an appropriate message is displayed.

Purpose: This function creates a "Detect Emotion" button that, when clicked, triggers the emotion detection process using the selected image.

Steps:
File Dialog: Opens a dialog to select an image file.
Thumbnail Creation: Resizes the selected image to fit within the GUI window.
Display Image: The selected image is displayed in the GUI.
Show Button: Calls show_Detect_button() to display the button for detecting emotions.

Buttons and Labels:

Upload Button: Allows users to upload an image.
Sign Image Label: Displays the uploaded image.
Label1: Displays the detected emotion.
Heading: Displays the title "Emotion Detector" at the top of the GUI.
Mainloop: The mainloop() function starts the Tkinter event loop, making the GUI responsive and interactive.

"""