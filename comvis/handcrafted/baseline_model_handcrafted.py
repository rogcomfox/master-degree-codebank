import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import os
from imutils import paths
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def init_data(path_to_train):
    img_list = list(paths.list_images(path_to_train))
    categories = ['Truck', 'Van']
    img_arr = []
    label_arr = []
    for i in categories:
        path = os.path.join(path_to_train, i)
        count = 0
        for img_path in os.listdir(path):
            img = cv.imread(os.path.join(path, img_path))
            retval = cv.haveImageReader(os.path.join(path, img_path))
            if retval == True:    
                img = cv.resize(img, (224, 224))
                img_arr.append(img.flatten())
                label_arr.append(categories.index(i))
                count += 1
            else:
                os.remove(img_path)
                # print data reading of image
            print("Processing {} images from total {} images".format(count, len(img_list)))
    img_data = np.array(img_arr)
    label_data = np.array(label_arr)
    #define ram consumption
    print('Ram Consumption: ', (img_data.nbytes / (1024 * 1000.0)))
    # define train data 
    x_train, x_test, y_train, y_test = train_test_split(img_data, label_data, test_size=0.15, random_state=42)
    # building model
    print("Evaluate Accuracy....")
    svm_model = svm.SVC(probability=True, C=1,  class_weight={1: 10}, gamma=0.0001, kernel='linear')
    svm_model.fit(x_train, y_train)
    #create accuracy
    y_pred = svm_model.predict(x_test)
    print('SVM Accuracy: {} %'.format(accuracy_score(y_pred, y_test)*100))
    # save model
    pickle.dump(svm_model, open('svm_model.p', 'wb'))
    # start = time.time()
    # knn_model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    # knn_model.fit(x_train, y_train)
    # end = time.time()
    # y_pred = knn_model.predict(x_test)
    # print('KNN Accuracy: {} %'.format(knn_model.score(x_test, y_test)))
    # # print('KNN Accuracy: {} %'.format(accuracy_score(y_pred, y_test)*100))
    # pickle.dump(knn_model, open('knn_model.p', 'wb'))
    # print('End Training phase in: ', time.strftime('%H:%M:%S', time.gmtime(end - start)))
    # svm_model = svm.SVC(probability=True, C=10, gamma=0.0001, kernel='rbf')
    # svm_model.fit(x_train, y_train)
    # #create accuracy
    # y_pred = svm_model.predict(x_test)
    # print('SVM Accuracy: {} %'.format(accuracy_score(y_pred, y_test)*100))
    # # save model
    # pickle.dump(svm_model, open('svm_model.p', 'wb'))
    # return x_train, y_train, x_test, y_test

# def svm_model():
    