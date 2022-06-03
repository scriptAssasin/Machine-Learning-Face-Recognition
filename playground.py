import cv2
import zipfile
import numpy as np
import os
import shutil
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier


current_path = os.getcwd()
# file_name = "datas/faces.zip"

#  # opening the zip file in READ mode 
# with zipfile.ZipFile(file_name, 'r') as zip:

#  # printing all the contents of the zip file 
# #    zip.printdir()

#  # extracting all the files 
#    print('Extracting all the files now...') 
#    zip.extractall() 


# faces_dir = os.listdir( current_path + "/faces")

# print(type(faces_dir))

# shutil.make_archive("faces_final", 'zip', "faces")

# faces_persons = {1: [], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
# faces_person_2 = {}
# faces_person_3 = {}
# faces_person_4 = {}
# faces_person_5 = {}
# faces_person_6 = {}
# faces_person_7 = {}
# faces_person_8 = {}
# faces_person_9 = {}
# faces_person_10 = {}

# with zipfile.ZipFile("faces_final.zip") as facezip:
#     # facezip.printdir()
#     for filename in facezip.namelist():

#         with facezip.open(filename) as image:
#             person_no = filename.split('_')[0]
#             person_no = int(person_no.replace('person', '', 1))
#             print(person_no)
#             # If we extracted files from zip, we can use cv2.imread(filename) instead
#             faces_persons[filename] = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

# print(faces_person_1)

# for filename in faces_dir:
#     print(filename)

    # break

def loadImages(path,set_number):
    array = []
    persons = []

    for filename in os.listdir(current_path + "/faces_sets_final/Set_1"):
        image = Image.open(current_path + "/faces_sets_final/Set_1/" + filename)
        np_image = np.array(image)
        flat_arr = np_image.ravel()
        print(type(flat_arr))
        array.append(flat_arr)
        print(np_image.shape)

        print(flat_arr.shape)

        person_no = filename.split('_')[0]
        person_no = int(person_no.replace('person', '', 1))
        persons.append(person_no)

    return array, persons

X_train, Y_train = loadImages(1,1)
X_test, Y_test = loadImages(1,1)

print(X_train, Y_train)

pca = PCA(n_components=30).fit(X_train)
X_train_pca = pca.transform(X_train)

classifier = KNeighborsClassifier().fit(X_train_pca, Y_train)

X_test_pca = pca.transform(X_test)
predictions = classifier.predict(X_test_pca)

print(classification_report(Y_test, predictions))


# print(array)