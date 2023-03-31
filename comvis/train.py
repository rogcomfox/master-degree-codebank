# for training data
import torch # PyTorch package
import torchvision.datasets as datasets # load datasets
import torchvision.models as models
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.optim as optim # optimzer
import torch.utils.data as load_data
import torch.nn.functional as F
from tqdm import tqdm
# import backbone.main_net as net

def train(path_to_train_data, batch_size=4, num_workers=2, num_epochs=50):
    data_transform = transforms.Compose(
        [transforms.RandomResizedCrop(256),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    #load train data
    trainset = datasets.ImageFolder(root=path_to_train_data, transform=data_transform)
    trainloader = load_data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    #define classess
    classes = ('Bicycle', 'Bus', 'Boat', 'Car', 'Helicopter', 'Motorcycle', 'Taxi', 'Truck', 'Van')

    #loss function
    # criterion = nn.CrossEntropyLoss() #based on paper
    # main_net = net.MainNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    model.cuda()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.01, weight_decay=1e-3, momentum=0.9) #based on paper

    model.train()
    # model.train()
    for epoch in range(num_epochs):
        with tqdm(trainloader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                loss = F.nll_loss(output, target)
                correct = (predictions == target).sum().item()
                accuracy = correct / batch_size

                loss.backward()
                optimizer.step()
                
                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
            
    #save model
    torch.save(model, './new_model.pth')
