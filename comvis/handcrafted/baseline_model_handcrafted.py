import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime
import os
import pandas as pd
import json
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

#global var
categories = ['Bicycle', 'Bus', 'Car', 'Motorcycle', 'Truck', 'Van']
# categories = ['Ambulance', 'Segway'] #this only fo test code purpose, not the whole code

#load data
def init_data(path_to_train, is_pca=True):
    # img_list = list(paths.list_images(path_to_train))
    #full categories = Ambulance, Barge, Bicycle, Boat, Bus, Car, Cart, Caterpillar, Helicopter, Limousine, Motorcycle, Segway, Snowmobile, Tank, Taxi, Truck, Van
    # categories = ['Ambulance', 'Barge', 'Bicycle', 'Boat', 'Bus', 'Car', 'Cart', 'Caterpillar', 'Helicopter', 'Limousine', 'Motorcycle', 'Segway', 'Snowmobile', 'Tank', 'Taxi', 'Truck', 'Van']
    img_arr = []
    label_arr = []
    print("Start Read Image Data if PCA = {} at {}".format(is_pca, datetime.datetime.now()))
    for i in categories:
        path = os.path.join(path_to_train, i)
        for img_path in os.listdir(path):
            img = cv.imread(os.path.join(path, img_path))
            retval = cv.haveImageReader(os.path.join(path, img_path))
            if retval == True:    
                img = cv.resize(img, (224, 224))
                if is_pca:
                    img_pca = pca(img)
                    img_arr.append(img_pca.flatten())
                else:
                    img_arr.append(img.flatten())
                label_arr.append(categories.index(i))
            else:
                os.remove(img_path)
                # print data reading of image
        print('Done Processing Category : ', i)
        # print("Processing {} images from total {} images".format(count, len(img_list)))
    print("Finish Read Image Data if PCA = {} at {}".format(is_pca, datetime.datetime.now()))
    img_data = np.array(img_arr)
    label_data = np.array(label_arr)
    df = pd.DataFrame(img_data)
    df['target'] = label_data
    img_final_data = df.iloc[:,:-1]
    label_final_data = df.iloc[:,-1]
    #define ram consumption
    print('Ram Consumption: ', (img_data.nbytes / (1024 * 1000.0)))
    ram = img_data.nbytes / (1024 * 1000.0)
    return is_pca, img_final_data, label_final_data, ram

#feature extraction with pca for reduce dimension (default: 100)
def pca(img, p_components=100):
    b, g, r = cv.split(img)
    init_pca = PCA(p_components)
    #transfrom and inverse red channel
    r_transformed = init_pca.fit_transform(r)
    r_inverted = init_pca.inverse_transform(r_transformed)
    #transform and inverse green channel
    g_transformed = init_pca.fit_transform(g)
    g_inverted = init_pca.inverse_transform(g_transformed)
    #transform and inverse blue channel
    b_transformed = init_pca.fit_transform(b)
    b_inverted = init_pca.inverse_transform(b_transformed)
    #stack channel
    img_compressed = (np.dstack((r_inverted, g_inverted, b_inverted))).astype(np.uint8)
    return img_compressed

#classify with svm (tidak dipakai) TODO: TIDAK DIPAKAI, KENDALA PROSES LAMA
def svm(img_arr, label_arr, is_pca, ram, c=1, gamma=0.0001, kernel='linear'):
    (x_train, x_test, y_train, y_test) = train_test_split(img_arr, label_arr, test_size=0.2, random_state=42)
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

def naive_bayes(img_arr, label_arr, is_pca, ram):
    (x_train, x_test, y_train, y_test) = train_test_split(img_arr, label_arr, test_size=0.2, random_state=42)
    #building model
    print("Evaluate Accuracy with Naive Bayes....")
    start_time = datetime.datetime.now()
    print('Start Training: ', start_time)
    naive_model = GaussianNB()
    naive_model.fit(x_train, y_train)
    y_pred = naive_model.predict(x_test)
    naive_acc = accuracy_score(y_pred, y_test)*100
    print('Naive Bayes Accuracy: {} %'.format(round(naive_acc, 3)))
    # save model
    if is_pca:
        pickle.dump(naive_model, open('naive_model_is_pca.p', 'wb'))
    else:
        pickle.dump(naive_model, open('naive_model.p', 'wb'))
    end_time = datetime.datetime.now()
    print('Finish Training Naive Bayes: ', end_time)
    arr_log = ["NaiveBayes", is_pca, str(start_time), str(end_time), str(naive_acc), str(ram)]
    create_log_file(arr_log)
    show_report(y_pred, y_test)

#classify with knn
def knn(img_arr, label_arr, is_pca, ram, n_neighbors=2, n_jobs=-1):
    (x_train, x_test, y_train, y_test) = train_test_split(img_arr, label_arr, test_size=0.2, random_state=42)
    #building model
    print("Evaluate Accuracy with KNN....")
    start_time = datetime.datetime.now()
    print('Start Training: ', start_time)
    # lix =[]
    # liy = []
    # idx = 0
    # acc = 0
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
    knn_model.fit(x_train, y_train)
    # for k in range(1, n_neighbors+1):
    #     liy.append(knn_model.score(x_test, y_test))
    #     if liy[k-1] > acc:
    #         acc = liy[k-1]
    #         idx = k-1
    #     lix.append(k)
    # print('KNN Accuracy (Best) at k={} with acc={}%'.format(str(idx+1), str(acc*100)))
    y_pred = knn_model.predict(x_test)
    knn_acc = accuracy_score(y_pred, y_test)*100
    print('KNN Accuracy: {} %'.format(round(knn_acc, 3)))
    # save model
    if is_pca:
        pickle.dump(knn_model, open('knn_model_is_pca.p', 'wb'))
    else:
        pickle.dump(knn_model, open('knn_model.p', 'wb'))
    end_time = datetime.datetime.now()
    print('Finish Training: ', end_time)
    #create log file
    arr_log = ["KNN", is_pca, str(start_time), str(end_time), str(knn_acc), str(ram)]
    create_log_file(arr_log)
    show_report(y_pred, y_test)

def create_log_file(arr_log):
    #plot data
    dict = {
        "Model Name": arr_log[0],
        "Using PCA": arr_log[1],
        "Start Training Time": arr_log[2],
        "Finish Training Time": arr_log[3],
        "Accuracy": arr_log[4],
        "RAM Consumption": arr_log[5]
    }
    json_obj = json.dumps(dict, indent=4)
    with open(f'{arr_log[0]}_PCA_{arr_log[1]}_log.json', 'w') as out:
        out.write(json_obj)

def test_model(model_path, path_to_single_test):
    model = pickle.load(open(model_path, 'rb'))
    img_test = cv.imread(path_to_single_test)
    img_resize = cv.resize(img_test, (224,224))
    plt.imshow(img_resize)
    plt.show()
    img_arr = [img_resize.flatten()]
    predict = model.predict_proba(img_arr)
    for ind,val in enumerate(categories):
        print(f'{val} = {round(predict[0][ind]*100,3)}%')
    print("The predicted image is : ", categories[model.predict(img_arr)[0]])

# eval the model
def show_report(y_pred, y_test):
    print(classification_report(y_true=y_test, y_pred=y_pred, target_names=categories))