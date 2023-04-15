from baseline_model_handcrafted import init_data, knn, test_model, naive_bayes

is_pca, img_arr, label_arr, ram = init_data('C://Users/ROG_is_Awesome/Downloads/vehicle_copy/train/train/', False)
print(img_arr)
print(label_arr)
# naive_bayes(img_arr, label_arr, is_pca, ram)
# if is_pca:
#     test_model('naive_model_is_pca.p', 'D://dataset-viskom/test/testset/000024.jpg')
# else:
#     test_model('naive_model.p', 'D://dataset-viskom/test/testset/000024.jpg')

# test_model('naive_model.p', 'D://dataset-viskom/test/testset/000024.jpg')