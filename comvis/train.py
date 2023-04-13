import torch
import numpy as np
import time
import os
import copy
from typing import Union
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from util.img_transform import ImgTransform
from imutils import paths

def train_model(
    model, 
    dataloaders, 
    criterion, 
    optimizer, 
    device: torch.device,
    save_best_model_path: Union[None, str] = None,
    num_epochs: int = 25, 
    is_inception: bool = False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, num_epochs+1):
        print()
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            seen_images = 0
            current_accuracies_pbar = []
            current_loss_pbar = []

            # Iterate over data.
            progress_bar = tqdm(dataloaders[phase], desc=f'{phase}: ({epoch}/{num_epochs})')
            for i, (inputs, labels) in enumerate(progress_bar):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # add loss to the progress bar
                    current_accuracies_pbar.append(torch.sum(preds == labels.data).item() / len(inputs))
                    current_loss_pbar.append(loss.item())
                    if i % 10 == 0:
                        desc = f'{phase} ({epoch}/{num_epochs}): '
                        desc += f'Loss: {np.mean(current_loss_pbar):.5f}; '
                        desc += f'Acc: {np.mean(current_accuracies_pbar):.5f}'

                        progress_bar.set_description(desc)
                        current_accuracies_pbar = []
                        current_loss_pbar = []

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                seen_images += len(labels)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / seen_images
            epoch_acc = running_corrects.double() / seen_images

            print(f'{phase} Loss: {epoch_loss:.5f} Acc: {epoch_acc:.5f}')

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
            if phase == 'valid':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val (@ {best_epoch}/{num_epochs} epoch): acc: {best_acc:4f}; loss: {best_loss:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    # save best model
    if save_best_model_path is not None:
        os.makedirs(os.path.split(save_best_model_path)[0], exist_ok=True)
        
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
            'accuracy': best_acc
        }, save_best_model_path)

    # train extra epochs on 

    return model, val_acc_history, best_loss, best_acc