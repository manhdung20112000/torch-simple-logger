from config import *
import torch
import torchvision
import torchvision.transforms as transforms

def get_transform():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    return transform

def get_dataset(path, transform, download=True):
    train_set = torchvision.datasets.MNIST(root=path, train=True, download=download, transform=transform)
    test_set  = torchvision.datasets.MNIST(root=path, train=False, download=download, transform=transform)
    return train_set, test_set

def get_dataloader(train_set, test_set):
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=False)

    return trainloader, testloader
