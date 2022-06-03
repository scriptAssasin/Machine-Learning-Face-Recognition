# import cv2
import zipfile
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier


current_path = os.getcwd()
zip_path = "datas/faces.zip"

 # opening the zip file in READ mode 
with zipfile.ZipFile(zip_path, 'r') as zip:

   print('Extracting all images now...') 
   zip.extractall() 


def loadImages(path,set_no):
    images_array = []
    persons = []

    if set_no == 1:
        for filename in os.listdir(path):
            image_no = int( filename.split("_")[1].split(".")[0] )

            if image_no >= 1 and image_no <= 7:
                image = Image.open(path + filename)
                np_image = np.array(image)
                flat_arr = np_image.ravel()
                images_array.append(flat_arr)
                person_no = filename.split('_')[0]
                person_no = int(person_no.replace('person', '', 1))
                persons.append(person_no)

    elif set_no == 2:
        for filename in os.listdir(path):
            image_no = int(filename.split("_")[1].split(".")[0])

            if image_no >= 8 and image_no <= 19:
                image = Image.open(path + filename)
                np_image = np.array(image)
                flat_arr = np_image.ravel()
                images_array.append(flat_arr)
                person_no = filename.split('_')[0]
                person_no = int(person_no.replace('person', '', 1))
                persons.append(person_no)

    elif set_no == 3:
        for filename in os.listdir(path):
            image_no = int(filename.split("_")[1].split(".")[0])

            if image_no >= 20 and image_no <= 31:
                image = Image.open(path + filename)
                np_image = np.array(image)
                flat_arr = np_image.ravel()
                images_array.append(flat_arr)
                person_no = filename.split('_')[0]
                person_no = int(person_no.replace('person', '', 1))
                persons.append(person_no)

    elif set_no == 4:
        for filename in os.listdir(path):
            image_no = int(filename.split("_")[1].split(".")[0])

            if image_no >= 32 and image_no <= 45:
                image = Image.open(path + filename)
                np_image = np.array(image)
                flat_arr = np_image.ravel()
                images_array.append(flat_arr)
                person_no = filename.split('_')[0]
                person_no = int(person_no.replace('person', '', 1))
                persons.append(person_no)

    elif set_no == 5:
        for filename in os.listdir(path):
            image_no = int(filename.split("_")[1].split(".")[0])

            if image_no >= 46 and image_no <= 64:
                image = Image.open(path + filename)
                np_image = np.array(image)
                flat_arr = np_image.ravel()
                images_array.append(flat_arr)
                person_no = filename.split('_')[0]
                person_no = int(person_no.replace('person', '', 1))
                persons.append(person_no)

    return images_array, persons

X_train, Y_train = loadImages(current_path + "/faces/",2)
X_test, Y_test = loadImages(current_path + "/faces/",2)

std = StandardScaler()

X_train = std.fit_transform(X_train)

X_test = std.fit_transform(X_test)

# print(X_train, Y_train)

pca = PCA(n_components=30).fit(X_train)
X_train_pca = pca.transform(X_train)

classifier = KNeighborsClassifier().fit(X_train_pca, Y_train)

X_test_pca = pca.transform(X_test)
predictions = classifier.predict(X_test_pca)

# print(classification_report(Y_test, predictions))


# print(array)