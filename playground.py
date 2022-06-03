# import cv2
import zipfile
import numpy as np
import os
import matplotlib.pyplot as plt

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


def loadImages(path,set_number):
    images_array = []
    persons = []

    if set_number == 1:
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

    elif set_number == 2:
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

    elif set_number == 3:
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

    elif set_number == 4:
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

    elif set_number == 5:
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

X_train, Y_train = loadImages(current_path + "/faces/",1)

X_test_1, Y_test_1 = loadImages(current_path + "/faces/",1)
X_test_2, Y_test_2 = loadImages(current_path + "/faces/",2)
X_test_3, Y_test_3 = loadImages(current_path + "/faces/",3)
X_test_4, Y_test_4 = loadImages(current_path + "/faces/",4)
X_test_5, Y_test_5 = loadImages(current_path + "/faces/",5)



std = StandardScaler()

X_train = std.fit_transform(X_train)

X_test_1 = std.fit_transform(X_test_1)
X_test_2 = std.fit_transform(X_test_2)
X_test_3 = std.fit_transform(X_test_3)
X_test_4 = std.fit_transform(X_test_4)
X_test_5 = std.fit_transform(X_test_5)

# print(X_train, Y_train)

pca = PCA(n_components=9).fit(X_train)
X_train_pca = pca.transform(X_train)

classifier = KNeighborsClassifier(metric="euclidean").fit(X_train_pca, Y_train)

X_test_1_pca = pca.transform(X_test_1)
predictions_1 = classifier.predict(X_test_1_pca)

X_test_2_pca = pca.transform(X_test_2)
predictions_2 = classifier.predict(X_test_2_pca)

X_test_3_pca = pca.transform(X_test_3)
predictions_3 = classifier.predict(X_test_3_pca)

X_test_4_pca = pca.transform(X_test_4)
predictions_4 = classifier.predict(X_test_4_pca)

X_test_5_pca = pca.transform(X_test_5)
predictions_5 = classifier.predict(X_test_5_pca)

print(classification_report(Y_test_1, predictions_1))


pca = PCA(n_components=30).fit(X_train)
X_train_pca = pca.transform(X_train)

classifier = KNeighborsClassifier(metric="euclidean").fit(X_train_pca, Y_train)

X_test_1_pca = pca.transform(X_test_1)
predictions_1 = classifier.predict(X_test_1_pca)

X_test_2_pca = pca.transform(X_test_2)
predictions_2 = classifier.predict(X_test_2_pca)

X_test_3_pca = pca.transform(X_test_3)
predictions_3 = classifier.predict(X_test_3_pca)

X_test_4_pca = pca.transform(X_test_4)
predictions_4 = classifier.predict(X_test_4_pca)

X_test_5_pca = pca.transform(X_test_5)
predictions_5 = classifier.predict(X_test_5_pca)

plt.imshow(pca.components_[0].reshape(50,50),cmap='binary_r')
plt.show()



# print(array)