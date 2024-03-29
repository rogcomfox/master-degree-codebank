import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
import os

from train import train_model
from test_model import test_model
from dataset.loader import get_train_valid_loader, get_test_loader
from backbone.main_net import initialize_model
# from util.util import workdir_copy
# from util.img_transform import ImgTransform
from datetime import datetime

#run main method for models based on NN
def main_network():
    
    #set max_split_size_mb in case out of memory 
    # checks and logs
    pwd = os.getcwd()
    print(f'Working dir: {pwd}')
    
    # fix the random seed
    seed = 13
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # paths to dataset
    train_data_dir = 'C://Users/ROG_is_Awesome/Downloads/vehicle/train/train/'
    test_data_dir = 'C://Users/ROG_is_Awesome/Downloads/vehicle/test/testset/'

    # Number of classes in the dataset
    num_classes = len(os.listdir(train_data_dir))

    model_name = "vgg"
    feature_extract = False
    # define the paths
    save_pred_path = None
    save_pred_path = f'result/{model_name}/pred_result{model_name}.csv'
    save_best_model_path = f'result/{model_name}/classify_model_{model_name}.pth'
    # hyper parameters
    device = torch.device("cuda:0")
    valid_size = 0.10
    lr = 1e-4
    batch_size = 64
    num_workers = 16
    pin_memory = True
    weighted_train_sampler = False
    weighted_loss = False
    num_epochs = 40

    torch.cuda.set_device(device)

    model_ft, input_size = initialize_model(
        model_name, num_classes, feature_extract, use_pretrained=True)

    # Data augmentation and normalization for training
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            # ImgTransform(input_size, 0.25),
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ]),
        'valid': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ]),
    }

    train_loader, valid_loader = get_train_valid_loader(
        train_data_dir, batch_size, data_transforms, seed, weighted_train_sampler,
        valid_size=valid_size, shuffle=True, show_sample=True, num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = get_test_loader(
        test_data_dir, batch_size, data_transforms, num_workers=num_workers, pin_memory=pin_memory)

    dataloaders_dict = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=lr)

    # Setup the loss fxn
    if weighted_loss:
        print('Weighted Loss')
        # {0: 0.010101, 1: 0.006622, 2: 0.0008244, 3: 0.00015335, 4: 0.0006253, 5: 0.00019665,
        # 6: 0.02631, 7: 0.00403, 8: 0.001996, 9: 0.01818, 10: 0.0004466, 11: 0.008771, 12: 0.01087,
        # 13: 0.006493, 14: 0.0017, 15: 0.000656, 16: 0.001200}
        cls_to_weight = train_loader.dataset.cls_to_weight
        weights = torch.FloatTensor([cls_to_weight[c] for c in range(num_classes)]).to(device)
    else:
        weights = torch.FloatTensor([1.0 for c in range(num_classes)]).to(device)

    criterion = nn.CrossEntropyLoss(weights)

    # print some things here so it will be seen in terminal for longer time
    print(f'Start Training Time: {datetime.now()}')
    print(f'using model: {model_name}')
    print(f'Using optimizer: {optimizer_ft}')
    print(f'Device {device}')
    print(f'Batchsize: {batch_size}')
    print(f'Transforms: {data_transforms}')

    # Train and evaluate
    model_ft, hist, loss, acc = train_model(
        model_ft, dataloaders_dict, criterion, optimizer_ft, device, save_best_model_path,
        num_epochs=num_epochs
    )

    # do test inference
    if save_pred_path is not None:
        test_model(model_ft, loss, acc, dataloaders_dict, device, save_pred_path)

if __name__ == "__main__":
     main_network()
     print("Finish Training Time: ", datetime.now())