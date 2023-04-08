import os
import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def init_data(path_to_train):
    categories=['Ambulance', 'Barge', 'Bicycle', 'Boat', 'Bus', 'Car', 'Cart', 'Caterpillar', 'Helicopter', 'Limousine', 'Motorcycle', 'Segway', 'Snowmobile', 'Tank', 'Taxi', 'Truck', 'Van']
    img_arr = []
    label_arr = []
    for i in categories:
        actual_path = os.path.join(path_to_train, i)
        for img in os.listdir(actual_path):
            img = cv.imread(os.path.join(actual_path, img))
            img = cv.resize(img, (224, 224))
            img_arr.append(img.flatten())
            label_arr.append(categories.index(i))
    img_data = np.array(img_arr)
    label_data = np.array(label_arr)
    dataframe = pd.DataFrame(img_data)
    dataframe['label'] = label_data

def svm_model():
    x = 0