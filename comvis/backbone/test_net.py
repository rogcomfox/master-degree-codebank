from torchsummary import summary
from torchvision import models

# model_ft = models.resnet18()
# model_ft = models.alexnet()
model_ft = models.vgg11_bn()
print(summary(model_ft.cuda(), input_size=(3, 224, 224)))