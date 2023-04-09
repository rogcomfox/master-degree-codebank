import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime
import os
import json
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#global var
categories = ['Bicycle', 'Bus', 'Car', 'Motorcycle', 'Truck', 'Van']

def init_data(path_to_train):
    # img_list = list(paths.list_images(path_to_train))
    #full categories = Ambulance, Barge, Bicycle, Boat, Bus, Car, Cart, Caterpillar, Helicopter, Limousine, Motorcycle, Segway, Snowmobile, Tank, Taxi, Truck, Van
    # categories = ['Ambulance', 'Barge', 'Bicycle', 'Boat', 'Bus', 'Car', 'Cart', 'Caterpillar', 'Helicopter', 'Limousine', 'Motorcycle', 'Segway', 'Snowmobile', 'Tank', 'Taxi', 'Truck', 'Van']
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
        print('Done Processing Category : ', i)
        # print("Processing {} images from total {} images".format(count, len(img_list)))
    img_data = np.array(img_arr)
    label_data = np.array(label_arr)
    #define ram consumption
    print('Ram Consumption: ', (img_data.nbytes / (1024 * 1000.0)))
    return img_data, label_data

def svm(img_arr, label_arr, c=1, gamma=0.0001, kernel='linear'):
    x_train, x_test, y_train, y_test = train_test_split(img_arr, label_arr, test_size=0.2, random_state=42)
    #building model
    print("Evaluate Accuracy with SVM....")
    print('Start Training SVM: ', datetime.datetime.now())
    svm_model = svm.SVC(probability=True, C=c,  class_weight={1: 10}, gamma=gamma, kernel=kernel)
    svm_model.fit(x_train, y_train)
    y_pred = svm_model.predict(x_test)
    svm_acc = accuracy_score(y_pred, y_test)*100
    print('SVM Accuracy: {} %'.format(round(svm_acc, 3)))
    # save model
    pickle.dump(svm_model, open('svm_model.p', 'wb'))
    print('Finish Training SVM: ', datetime.datetime.now())

def knn(img_arr, label_arr, n_neighbors=2, n_jobs=-1):
    x_train, x_test, y_train, y_test = train_test_split(img_arr, label_arr, test_size=0.2, random_state=42)
    #building model
    print("Evaluate Accuracy with KNN....")
    start_time = datetime.datetime.now()
    print('Start Training: ', start_time)
    lix =[]
    liy = []
    idx = 0
    acc = 0
    for k in range(1, n_neighbors+1):
        knn_model = KNeighborsClassifier(n_neighbors=k, n_jobs=n_jobs)
        knn_model.fit(x_train, y_train)
        liy.append(knn_model.score(x_test, y_test))
        if liy[k-1] > acc:
            acc = liy[k-1]
            idx = k-1
        lix.append(k)
    print('KNN Accuracy (Best) at k={} with acc={}%'.format(str(idx+1), str(acc*100)))
    y_pred = knn_model.predict(x_test)
    knn_acc = accuracy_score(y_pred, y_test)*100
    print('KNN Accuracy: {} %'.format(round(knn_acc, 3)))
    # save model
    pickle.dump(knn_model, open('knn_model.p', 'wb'))
    end_time = datetime.datetime.now()
    print('Finish Training: ', end_time)
    #create log file
    arr_log = ["KNN", str(start_time), str(end_time), str(knn_acc), str(img_arr.nbytes / (1024 * 1000.0))]
    create_log_file(arr_log)

def create_log_file(arr_log):
    #plot data
    dict = {
        "Model Name": arr_log[0],
        "Start Training Time": arr_log[1],
        "Finish Training Time": arr_log[2],
        "Accuracy": arr_log[3],
        "RAM Consumption": arr_log[4]
    }
    json_obj = json.dumps(dict, indent=4)
    with open(f'{arr_log[0]}_log.json', 'w') as out:
        out.write(json_obj)

def test_model(model_path, path_to_single_test):
    model = pickle.load(open(model_path, 'rb'))
    img_test = cv.imread(path_to_single_test)
    img_resize = cv.resize(img_test, (224,224))
    plt.imshow(img_resize.flatten())
    plt.show()
    img_arr = [img_resize]
    predict = model.predict_proba(img_arr)
    for ind,val in enumerate(categories):
        print(f'{val} = {predict[0][ind]*100}%')
    print("The predicted image is : ", categories[model.predict(img_arr)[0]])
