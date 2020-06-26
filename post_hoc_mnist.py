import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torchvision import models, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_mnist():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.2860), (0.3530))])  # Precalculated based on test dataset

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                 download=True, transform=transform)

    trainset, valset = torch.utils.data.random_split(trainset, [int(len(trainset)*0.7), int(len(trainset)*0.3)])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    return trainset, valset, testset, trainloader, valloader, testloader


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(400, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def add_bias(labels):
    ytrue = (labels % 2).type(torch.float32)  # {0,1}
    rand = 2*(torch.rand(4).to(device) > 0.01) - 1  # {-1,1}
    delta = (rand * (2*ytrue-1)).reshape(labels.size(0), 1, 1, 1)  # {-1,1}
    delta = ((delta+1) // 2)  # {0, 1}
    return ytrue, delta


def val_run(model, valloader, criterion):
    outputs = []
    valloss = 0.
    yval_trues = []
    valdetas = []
    with torch.no_grad():
        for valdata in valloader:
            valinputs, vallabels = valdata[0].to(device), valdata[1].to(device)
            yval_true, valdelta = add_bias(vallabels)
            valdetas.append(valdelta)
            valoutputs = model(valinputs + valdelta * torch.randn_like(valinputs)*0.4).squeeze(-1)
            outputs.append(torch.sigmoid(valoutputs))
            valloss += criterion(valoutputs, yval_true).item()
            yval_trues.append(yval_true)
    return outputs, valloss, yval_trues, valdetas


def compute_bias(y_pred, y_true, priv, metric):
    def zero_if_nan(x):
        return 0. if np.isnan(x) else x

    gtpr_priv = zero_if_nan(y_pred[priv * y_true == 1].mean())
    gfpr_priv = zero_if_nan(y_pred[priv * (1-y_true) == 1].mean())
    mean_priv = zero_if_nan(y_pred[priv == 1].mean())

    gtpr_unpriv = zero_if_nan(y_pred[(1-priv) * y_true == 1].mean())
    gfpr_unpriv = zero_if_nan(y_pred[(1-priv) * (1-y_true) == 1].mean())
    mean_unpriv = zero_if_nan(y_pred[(1-priv) == 1].mean())

    if metric == "spd":
        return mean_unpriv - mean_priv
    elif metric == "aod":
        return 0.5 * ((gfpr_unpriv - gfpr_priv) + (gtpr_unpriv - gtpr_priv))
    elif metric == "eod":
        return gtpr_unpriv - gtpr_priv


def train_model(net, trainloader, valloader, criterion, optimizer):
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            ytrue, delta = add_bias(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs + delta * torch.randn_like(inputs)*0.5).squeeze(-1)
            loss = criterion(outputs, ytrue)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                net.eval()
                _, valloss, _, _ = val_run(net, valloader, criterion)
                print(f'[{epoch + 1},{i + 1}] trainloss: {running_loss / len(trainloader):.3f}, valloss: {valloss / len(valloader):.3f}')
                running_loss = 0.0


def main():
    trainset, valset, tetstset, trainloader, valloader, testloader = load_mnist()
    net = Model()
    net.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters())
    train_model(net, trainloader, valloader, criterion, optimizer)

    y_test = []
    ypred_test = []
    deltas = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            ytrue, delta = add_bias(labels)
            outputs = net(images + delta * torch.randn_like(images)*0.4).squeeze(-1)
            y_test.append(ytrue)
            deltas.append(delta)
            ypred_test.append(torch.sigmoid(outputs))

    y_test = torch.cat(y_test).cpu().numpy()
    print(y_test)

    deltas = torch.cat(deltas).squeeze().cpu().numpy()
    print(deltas)

    ypred_test = torch.cat(ypred_test).cpu().numpy()
    print(ypred_test)

    print((y_test == deltas).mean().item())

    fpr, tpr, thresholds = roc_curve(y_test, ypred_test)
    plt.plot(fpr, tpr)
    print(roc_auc_score(y_test, ypred_test))

    threshs = np.linspace(0, 1, 1001)
    best_thresh = np.max([accuracy_score(y_test, ypred_test > thresh) for thresh in threshs])
    accuracy_score(y_test, ypred_test > best_thresh)

    print(abs(compute_bias(ypred_test > best_thresh, y_test, deltas, 'aod')))


if __name__ == "__main__":
    main()
