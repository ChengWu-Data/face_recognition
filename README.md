Face Detection and Face Recognition Project


This project contains two parts. The first part is face detection, and the second part is face recognition. The goal of the project is to first extract Ariel Sharon’s face images from the raw image folder by using the provided Haar cascade model, and then perform face recognition on six classes by using the Eigenfaces method.

The six classes used in this project are:
ariel_sharon
chris_evans
chris_hemsworth
mark_ruffalo
robert_downey_jr
scarlett_johansson


1. Logic of the Code

The program is written in a single Python script named UNI_faces.py. The script is placed at the same level as the Faces folder, so all paths in the code are relative paths instead of absolute paths.

The whole program follows the order below.

First, the script checks whether the Faces folder exists and whether the file haarcascade_frontalface_default.xml exists. These two checks are necessary because the project cannot run correctly without the dataset folder and the provided face detector.

After that, the program searches for Ariel Sharon’s original image folder. In my dataset, the folder is named ariel_sharon_raw, so I wrote the code in a way that can recognize slightly different folder naming styles. This makes the script more stable and avoids failure caused only by naming differences.

Once the Ariel Sharon raw folder is found, the program uses the provided OpenCV Haar cascade file, haarcascade_frontalface_default.xml, to detect faces in each raw image. Every image is first converted to grayscale because Haar cascade detection works on grayscale images. If more than one face is detected in one image, the program keeps the largest detected face, since that is usually the main subject in the image.

After detection, the face region is cropped from the image and resized to a fixed size of 100 by 100 pixels. This step is important because the recognition model needs all images to have the same shape. The cropped face images are then saved into the folder named ariel_sharon by hands. This folder is used as Ariel Sharon’s final class folder for the recognition task.

To keep the output naming style consistent with the other character folders, the program also inspects one existing Avenger folder and uses it as a reference when building filenames for the cropped Ariel Sharon images.

After the face detection part is finished, the program starts the face recognition part. At this stage, the code loads all six required classes. For Ariel Sharon, it does not use the raw folder anymore. Instead, it uses the cropped folder ariel_sharon by hands. For the other five characters, it directly loads their existing face image folders.

Each face image is read in grayscale, resized to 100 by 100, and then flattened into a one-dimensional vector. These vectors are collected into the feature matrix X, and their class names are stored in the label array y.

Then the dataset is split into training data and testing data using test_size = 0.3 and random_state = 42, exactly as required in the assignment. The training set is the only part used to fit the model. The test set is kept separate and is only used for evaluation.

Before applying Eigenfaces, the code standardizes the feature values by fitting a StandardScaler on the training set and then applying that same transformation to both the training set and the test set. This avoids information leakage from the test data.

Next, the Eigenfaces method is implemented by PCA. PCA is fitted only on the training set. This means the principal components, which represent the Eigenfaces, are learned from the training images only. Both training and testing images are then projected into this lower-dimensional Eigenfaces space.

After that, the recognition model is trained on the transformed training data. In my final version, I used an SVC classifier after PCA. The Eigenfaces part of the method is still PCA, and the classifier is then used to separate the six classes in the reduced feature space.

Finally, the code predicts the labels of the test set and prints the recognition accuracy. In my final run, the program successfully saved 49 cropped Ariel Sharon face images and achieved an accuracy of 81.11 percent, which is higher than the 60 percent requirement for full credit.


2. Problems I Found

While working on this assignment, I found several practical problems.

The first problem was folder naming inconsistency. The assignment description refers to Ariel Sharon and the six required classes in one naming style, but the actual folders in the dataset may not always use exactly the same style. For example, some names may use spaces, some may use underscores, and Ariel Sharon’s raw folder in my dataset was named ariel_sharon_raw. At the beginning, this caused the program to fail because the code could not find the expected folder. I solved this by normalizing folder names before comparing them and by writing a separate function to find Ariel Sharon’s raw folder more flexibly.

The second problem was that Haar cascade face detection is not perfect. It works well for many images, but it can still miss faces in difficult cases, especially when the face is small, rotated, or partially blocked. In some images, more than one face-like region may be detected. To make the program behave more consistently, I chose the largest detected face in each image. This is a simple rule, but in this dataset it worked reasonably well.

The third problem was related to the recognition stage. My earlier version used a KNN classifier after PCA. Although the overall logic was correct, I ran into an environment-related error on my computer when calling the KNN prediction step. The issue came from the local Python package environment instead of the project logic itself. To make the program run successfully and stably, I replaced the classifier with SVC. The Eigenfaces part of the project still comes from PCA, so the overall recognition approach remains consistent with the assignment goal.

The fourth problem was keeping the saved Ariel Sharon filenames consistent with the other class folders. Since the assignment specifically asks for the segmented images to follow the same format as the other five Avenger character folders, I added code that tries to infer the file naming pattern from an existing folder. This made the output more systematic, although in practice I think this part still depends somewhat on how the original dataset was organized.

Another issue is that image quality directly affects recognition performance. Some images are clearer and more centered than others. Since Eigenfaces is a relatively classical method, it is sensitive to differences in lighting, pose, and background. This means the final accuracy does not only depend on the algorithm, but also on how clean and consistent the dataset is.


3. How the Model Can Be Improved

There are several ways this project could be improved.

The first improvement would be better face detection. The assignment requires Haar cascade, so I followed that instruction. However, if I were trying to improve the model beyond the assignment, I would replace Haar cascade with a stronger detector such as MTCNN or a Dlib-based detector. Those methods are usually more reliable when the face is not perfectly centered.

The second improvement would be face alignment before recognition. Right now, the program crops the detected face and resizes it, but it does not align the face based on eye position or head orientation. If face alignment were added, the images would become more consistent, and the recognition accuracy would likely improve.

The third improvement would be more tuning in the recognition step. In this version, PCA is used for Eigenfaces and SVC is used for classification, and the number of principal components is chosen in a simple way. I think it would be helpful to test different PCA dimensions and different classifier settings to see which combination gives the best performance on this dataset.

The fourth improvement would be better preprocessing of the cropped faces. For example, histogram equalization or other normalization methods could reduce lighting differences across images. Since Eigenfaces is sensitive to appearance variation, better preprocessing would probably improve stability.

The fifth improvement would be expanding the dataset or cleaning the dataset more carefully. If some images are poorly cropped, low quality, or visually inconsistent, they can hurt the recognition result. A cleaner and more balanced dataset would help the model generalize better.


4. Final Notes

This script uses only the training dataset for model fitting. The test dataset is used only for evaluation.

The data split is:
test_size = 0.3
random_state = 42

The script is written with relative paths, so it should be placed at the same level as the Faces folder.

The program requires the provided file haarcascade_frontalface_default.xml in the same directory as the Python script.

In my final successful run, the output was:

Cropped Ariel Sharon faces saved: 49
Accuracy: 81.11%
Training samples: 209
Testing samples: 90
Number of Eigenfaces: 80

This means the program ran successfully and the final recognition accuracy was above the required threshold.
