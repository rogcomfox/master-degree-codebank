# augmentation of iamge
import numpy as np
from imgaug import augmenters as imgaug

class ImgTransform(object):
    def __init__(self, model_input, param=0.25):
        self.model_input = model_input

        self.aug = imgaug.Sequential([
            imgaug.Sometimes(param, 
                imgaug.OneOf([
                    imgaug.MotionBlur(k=15, angle=[-135, -90, -45, 45, 90, 135]),
                    imgaug.GaussianBlur(sigma=(0, 3.0)),
                ])
            ),
            imgaug.Sometimes(param, imgaug.Affine(rotate=(-20, 20), mode='symmetric')),
            imgaug.Sometimes(param, imgaug.AddToHueAndSaturation(value=(-10, 10), per_channel=True)),
            imgaug.Sometimes(param, imgaug.ContrastNormalization((0.5, 1.5), per_channel=0.5)),
            imgaug.Sometimes(param, imgaug.Sharpen(alpha=0.5)),
            imgaug.Sometimes(param, imgaug.AdditiveGaussianNoise(scale=(0, 0.1*255))),
            imgaug.Sometimes(param, imgaug.Add((-20, 20), per_channel=0.5)),
            imgaug.Sometimes(param, imgaug.PiecewiseAffine(scale=(0.01, 0.02))),
            imgaug.Sometimes(param, imgaug.PerspectiveTransform(scale=(0.01, 0.15)))
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)