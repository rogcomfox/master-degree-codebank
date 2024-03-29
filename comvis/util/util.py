# for some setup
import matplotlib.pyplot as plt
import os
# from shutil import copytree, ignore_patterns
import numpy as np

from typing import Dict, Tuple


# def workdir_copy(pwd: str, copy_path: str):
#     cp_path = os.path.join(copy_path, 'wdir_copy')
#     copytree(pwd, cp_path, ignore=ignore_patterns('__pycache__', '.git'))

def save_predictions(
        save_path: str,
        res_loss,
        res_acc,
        predictions: Tuple[str, int],
        idx_to_class: Dict[int, str]
    ) -> None:
    '''
        Format:

        Train Accuracy:
        Val Accuray:
        Id,Category,ImgPath
        0,Car,bla.jpg
        1,Catepillar,bla.jpg
    '''
    with open(save_path, 'w') as outf:
        # header
        outf.write('Accuracy: ', res_acc)
        outf.write('Loss: ', res_loss)
        outf.write('Id,Category,ImgPath\n')

        # other lines
        for (pred_path, pred_idx) in predictions:
            # extract Id from the filename
            Id = int(os.path.split(pred_path)[1].strip('.jpg'))
            outf.write(f'{Id},{idx_to_class[pred_idx]},{pred_path}\n')

    print(f'Wrote preds to {save_path}')


def plot_images(images, data_dir, cls_true, cls_pred=None):
    label_names = sorted(os.listdir(data_dir))
    fig, axes = plt.subplots(5, 8, figsize=(15, 10))

    for i, ax in enumerate(axes.flat):
        # plot img
        means = np.array([0.485, 0.456, 0.406])
        stds = np.array([0.229, 0.224, 0.225])

        inp = np.clip(images[i, :, :, :] * stds + means, 0, 1)
        ax.imshow(inp, interpolation='spline16')

        # show true & predicted classes
        cls_true_name = label_names[cls_true[i]]
        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = label_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(
                cls_true_name, cls_pred_name
            )
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show(block=False)
    plt.pause(10)
    plt.close()