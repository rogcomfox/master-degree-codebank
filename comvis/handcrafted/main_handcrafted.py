from baseline_model_handcrafted import init_data, knn, test_model, naive_bayes

is_pca, img_arr, label_arr  = init_data('C://Users/ROG_is_Awesome/Downloads/vehicle_copy/train/train/', False)
# img_arr, label_arr = init_data('D://dataset-viskom/train/train/')
knn(img_arr, label_arr, is_pca, 5)
# test_model('knn_model.p', 'C://Users/ROG_is_Awesome/Downloads/vehicle_copy/test/testset/000024.jpg')