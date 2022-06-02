import cv2
import zipfile
import numpy as np
import os

faces = {}
with zipfile.ZipFile("datas/faces.zip") as facezip:
    # facezip.printdir()
    for filename in facezip.namelist():
        print(filename)
        file_paths = get_all_file_paths(directory)
        with facezip.open(filename) as image:
            # If we extracted files from zip, we can use cv2.imread(filename) instead
            faces[filename] = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

# EROTIMA 2
# https://stats.stackexchange.com/questions/69205/how-to-derive-the-ridge-regression-solution
# https://stats.stackexchange.com/questions/367508/showing-that-ridge-regression-is-a-solution-to-the-following-optimization-proble
# https://suzyahyah.github.io/optimization/2018/07/20/Constrained-unconstrained-form-Ridge.html#:~:text=The%20familiar%20ridge%20regression%20objective,in%20scenarios%20of%20high%20collinearity.

# EROTIMA 3
# https://machinelearningmastery.com/face-recognition-using-principal-component-analysis/