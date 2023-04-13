from torchvision import models
from torchsummary import summary

# model_ft = models.resnet18()
# model_ft = models.alexnet()
model_ft = models.vgg11_bn()

print('VGG11')
print(summary(model_ft.cuda(), input_size=(3, 256, 256)))