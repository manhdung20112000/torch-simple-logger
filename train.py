from models.model import Net
from utils.utils import get_dataloader, get_dataset, get_transform

import torch
import torch.nn.functional as F
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

from config import *

# for logging
from utils.logger import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

def train(model, epoch, trainloader, optimizer, loss_function):
    model.train()
    running_loss = 0
    for i, (input, target) in enumerate(trainloader, 0):
        # zero the gradient
        optimizer.zero_grad()

        # forward + backpropagation + step
        predict = model(input)
        loss = loss_function(predict, target)
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item()

    total_loss = running_loss/len(trainloader.dataset)
    logger.add_scalar('train/avg_loss',total_loss, global_step=epoch)
    
    # wandb save as artifact
    # torch.onnx.export(model, input, RUN_NAME+'.onnx')
    # wandb.save(RUN_NAME+'.onnx')
    # trained_weight = wandb.Artifact("CNN", type="model", description="test")
    # trained_weight.add_file(RUN_NAME+'.onnx')
    # run.log_artifact(trained_weight)
    # pytorch save
    # torch.save(model.state_dict(), SAVE_PATH+'.pth')
    return 

def test(model, epoch, testloader):
    model.eval()
    test_loss = 0
    correct   = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(testloader):
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            predict = output.data.max(1, keepdim=True)[1]
            correct += predict.eq(target.view_as(predict)).sum().item()
    
    test_loss /= len(testloader)
    test_accuracy = 100. * correct / len(testloader.dataset)
    
    scalars = {'loss': test_loss,
        'accuracy': test_accuracy,}

    logger.add_scalars(main_tag='eval', tag_scalar_dict=scalars, global_step=epoch)
    return test_loss, test_accuracy

if __name__ == '__main__':
    # init wandb
    config = dict(
        learning_rate = LEARNING_RATE,
        momentum      = MOMENTUM,
        architecture  = ARCHITECTURE,
        dataset       = DATASET,
    )

    logger = SummaryWriter(log_dir="mlops-wandb-demo",config=config)
    # run = wandb.init(project="mlops-wandb-demo", tags=["dropout", "cnn"], config=config)
    # run.use_artifact('mnist:latest')
    # artifact = wandb.Artifact('mnist', type='dataset')
    # artifact.add_dir(DATA_PATH)
    # run.log_artifact(artifact)
    
    # get dataloader
    train_set, test_set = get_dataset(transform=get_transform())
    trainloader, testloader = get_dataloader(train_set=train_set, test_set=test_set)

    # create model
    model = Net()

    # define optimizer and loss function
    epochs = EPOCHS
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    optimizer     = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # training
    pb = tqdm(range(epochs))
    train_losses, test_losses, test_accuracy = [], [], []

    # wandb watch model
    logger.watch_model(model=model, criterion=loss_function, log='all', log_freq=10)
    
    for epoch in pb:
        train_loss = train(model, epoch, trainloader, optimizer, loss_function)
        train_losses.append(train_loss)
        print(epoch, train_loss)

        test_loss, test_acc = test(model, epoch, testloader)
        test_losses.append(test_loss)
        test_accuracy.append(test_acc)
        print(epoch, test_loss, test_acc)



