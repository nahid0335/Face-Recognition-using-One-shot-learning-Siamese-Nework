# Face-Recognition-using-One-shot-learning-Siamese-Nework
Face Recognition System<br>
There is 'Face recognition.ipynb' file which is the jupyternotebook.<br>
The model is build and train in jupyternotebook.<br>
'Data' folder contains the dataset for this model.
<br>
# Overview
This is a face Recognition system in python.<br>
The model use one-shot learning with siamese netork.<br>
For Visualiation purpose kivy is used.<br>
For fewer dataset, one-shot learning is used to face recognition.
<br>
# Specification
- Language : Python
- Model : One-shot learning with siamese network
- Paper : [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) (This paper is followed to build the model)
- Dataset : [Wild dataset](http://vis-www.cs.umass.edu/lfw/) (partial data is taken from here)

<br>

# Description
There is 8 step in the 'Face recognition.ipynb'.<br>

## At 2.1 Untar Labelled Faces in the Wild Dataset
Download the dataset and extrat the dataset in the same folder of .ipynb file.<br>
Then move them to negative folder under data folder
## At 2.2 Collect Positive and Anchor Classes
Using the webcam take some picture for anchor and positive class. At least take 500+ images.

## At 5.2 Establish Checkpoints
Create a folder named "training_checkpoints"

## At 8.1 Verification Function
create folder "application_data".<br>
Under this folder create 2 subfolder named 'verification_images' which will contain 50 positive and anchore image.<br>
You have to copy those image from positive and anchor folder.<br>
Another folder named 'input_image' which will contain one image taken from webcam in realtime.


# Kivy App
For visualization Kivy is used.<br>
Open the faceapp.py file in vs code.<br>
Then install kivy by using "pip install kivy[full] kivy_examples".<br>
Then run the file.

# OUTPUT
This is the initial stage and unvarified.<br>
<br>
![This is an image](https://github.com/nahid0335/Face-Recognition-using-One-shot-learning-Siamese-Nework/blob/master/images/11.PNG)
<br>
<br>
After click the register, kivy will take 50 picture and register you.
<br>
Then by clicking the verify button, if your face match with the registration image and pass a certain thresshold then you will be verified.
<br>
<br>
![This is an image](https://github.com/nahid0335/Face-Recognition-using-One-shot-learning-Siamese-Nework/blob/master/images/13.PNG)
<br>
<br>
if you are different person then you will be unverified.
<br>
<br>
![This is an image](https://github.com/nahid0335/Face-Recognition-using-One-shot-learning-Siamese-Nework/blob/master/images/12.PNG)
<br>




