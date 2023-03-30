# for training data
import torch # PyTorch package
import torchvision.datasets as datasets # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.optim as optim # optimzer
import torch.utils.data as load_data
import backbone.main_net as net

def train(path_to_train_data, batch_size=4, num_workers=2, num_epochs=50):
    data_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    #load train data
    trainset = datasets.ImageFolder(root=path_to_train_data, transform=data_transform)
    trainloader = load_data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    #define classess
    classes = ('Bicycle', 'Bus', 'Boat', 'Car', 'Helicopter', 'Motorcycle', 'Taxi', 'Truck', 'Van')

    #loss function
    criterion = nn.CrossEntropyLoss() #based on paper
    main_net = net.MainNet()
    optimizer = optim.SGD(main_net.parameters(), lr=0.01, weight_decay=1e-3, momentum=0.9) #based on paper

    # start training
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    for epoch in range(num_epochs):  # loop over the dataset for 50 times (based on paper)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    # whatever you are timing goes here
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print('Finished Training')
    print(start.elapsed_time(end))  # milliseconds

    #save model
    torch.save(main_net.state_dict(), './new_model.pth')
