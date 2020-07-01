import copy
import math
import argparse

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
torch.manual_seed(0)

descriptions = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
                'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
                'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',
                'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie',
                'Young']

attractive_index = descriptions.index('Attractive')
male_index = descriptions.index('Male')


def load_celeba(input_size=224, num_workers=2, trainsize=100, testsize=100):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainset = torchvision.datasets.CelebA(root='./data', download=True, split='train', transform=transform)
    testset = torchvision.datasets.CelebA(root='./data', split='test', download=True, transform=transform)

    if trainsize >= 0:
        # cut down the training set
        trainset, _ = torch.utils.data.random_split(trainset, [trainsize, len(trainset) - trainsize])
    trainset, valset = torch.utils.data.random_split(trainset, [int(len(trainset)*0.7), int(len(trainset)*0.3)])
    if testsize >= 0:
        testset, _ = torch.utils.data.random_split(testset, [testsize, len(testset) - testsize])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=num_workers)

    return trainset, valset, testset, trainloader, valloader, testloader


def get_resnet_model():
    resnet18 = models.resnet18(pretrained=True)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 2)
    resnet18.to(device)
    return resnet18


def get_best_accuracy(y_true, y_pred, y_prot):
    threshs = torch.linspace(0, 1, 1001)
    best_acc, best_thresh = 0., 0.
    for thresh in threshs:
        acc = torch.mean(((y_pred > thresh) == y_true).float()).item()
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh
    return best_acc, best_thresh


def train_model(model, trainloader, valloader, criterion, optimizer, epochs=2):
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)

        model.train()

        running_loss = 0.
        running_corrects = 0

        for index, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), (labels[:, attractive_index]).float().to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs[:, 0], labels)

            preds = torch.sigmoid(outputs[:, 0]) > 0.5

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if (index-1) % 101 == 0:
                num_examples = index * inputs.size(0)
                print(f"({index}/{len(trainloader)}) Loss: {running_loss / num_examples:.4f} Acc: {running_corrects.float() / num_examples:.4f}")

        best_acc, _ = val_model(model, valloader, get_best_accuracy)
        print(f"Best Accuracy on Validation set: {best_acc}")


def val_model(model, loader, criterion):
    y_true, y_pred, y_prot = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels, protected = inputs.to(device), labels[:, attractive_index].float().to(device), labels[:, male_index].float().to(device)
            y_true.append(labels)
            y_prot.append(protected)
            y_pred.append(torch.sigmoid(model(inputs)[:, 0]))
    y_true, y_pred, y_prot = torch.cat(y_true), torch.cat(y_pred), torch.cat(y_prot)
    return criterion(y_true, y_pred, y_prot)


def val_run(model, valloader, criterion):
    outputs = []
    valloss = 0.
    yval_trues = []
    protected = []
    with torch.no_grad():
        for valdata in valloader:
            valinputs, vallabels = valdata[0].to(device), valdata[1].to(device)
            yval_true = get_single_attr(vallabels)
            protected_label = get_single_attr(vallabels, attr='Male')
            valoutputs = model(valinputs).squeeze(-1)
            outputs.append(torch.sigmoid(valoutputs))
            valloss += criterion(valoutputs, yval_true).item()
            yval_trues.append(yval_true)
            protected.append(protected_label)
    return outputs, valloss, yval_trues, protected


def get_single_attr(labels, attr='Attractive'):

    # print(labels.shape)
    newlabels = []
    for i in range(len(labels)):
        newlabels.append(labels[i][descriptions.index(attr)])
    newlabels = torch.from_numpy(np.array(newlabels))
    newlabels = newlabels.float()
    # print(newlabels.shape)
    return newlabels


def compute_priors(data):
    counts = np.array([[0, 0], [0, 0]])
    for batch in list(data):
        imgs, labels = batch[0], batch[1]

        for label in labels:
            pro_value = label[male_index]
            attr_value = label[attractive_index]
            counts[pro_value][attr_value] += 1
    total = sum(sum(counts))
    protected_rate = np.round(counts[1][1]/sum(counts[1]), 4)
    unprotected_rate = np.round(counts[0][1]/sum(counts[0]), 4)

    print('Prob. Male:', np.round(sum(counts[1])/total, 4))
    print('Prob. Attractive:', np.round(sum(counts[:, 1])/total, 4))
    print('Prob. Attractive given Male', protected_rate)
    print('Prob. Attractive given Female', unprotected_rate)


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


def main(args):

    trainsize = args.trainsize
    testsize = args.testsize
    num_workers = args.num_workers
    print_priors = args.print_priors

    trainset, valset, testset, trainloader, valloader, testloader = load_celeba(trainsize=trainsize,
                                                                                testsize=testsize,
                                                                                num_workers=num_workers)

    if print_priors:
        compute_priors(testloader)

    net = get_resnet_model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters())
    train_model(net, trainloader, valloader, criterion, optimizer)

    def get_objective_with_best_accuracy(y_true, y_pred, y_prot):
        rocauc_score = roc_auc_score(y_true.cpu(), y_pred.cpu())
        best_acc, best_thresh = get_best_accuracy(y_true, y_pred, y_prot)
        bias = compute_bias((y_pred > best_thresh).float().cpu(), y_true.float().cpu(), y_prot.float().cpu(), 'aod')
        obj = .75*abs(bias)+(1-.75)*(1-best_acc)
        return rocauc_score, best_acc, bias, obj

    rocauc_score, best_acc, bias, obj = val_model(net, testloader, get_objective_with_best_accuracy)

    print('roc auc', rocauc_score)
    print('accuracy with best thresh', best_acc)
    print('aod', bias.item())
    print('objective', obj.item())

    def get_best_objective(y_true, y_pred, y_prot):
        threshs = torch.linspace(0, 1, 501)
        best_obj, best_thresh = math.inf, 0.
        for thresh in threshs:
            acc = torch.mean(((y_pred > thresh) == y_true).float()).item()
            bias = compute_bias((y_pred > thresh).float().cpu(), y_true.float().cpu(), y_prot.float().cpu(), 'aod')
            obj = .75*abs(bias)+(1-.75)*(1-best_acc)
            if obj < best_obj:
                best_obj, best_thresh = obj, thresh
        return best_obj, best_thresh

    rand_result = [math.inf, None, -1]
    rand_model = copy.deepcopy(net)
    for iteration in range(101):
        rand_model.to(device)
        for param in rand_model.parameters():
            param.data = param.data * (torch.randn_like(param) * 0.1 + 1)

        rand_model.eval()
        best_obj, best_thresh = val_model(rand_model, valloader, get_best_objective)
        if best_obj < rand_result[0]:
            del rand_result[1]
            rand_result = [best_obj, rand_model.state_dict(), best_thresh]

        if iteration % 10 == 0:
            print(f"{iteration} / 101 trials have been sampled.")

    # evaluate best random model
    best_model = copy.deepcopy(net)
    best_model.load_state_dict(rand_result[1])
    best_model.to(device)
    best_thresh = rand_result[2]

    def get_best_objective_results(best_thresh):
        def _get_results(y_true, y_pred, y_prot):
            rocauc_score = roc_auc_score(y_true.cpu(), y_pred.cpu())
            acc = torch.mean(((y_pred > best_thresh) == y_true).float()).item()
            bias = compute_bias((y_pred > best_thresh).float().cpu(), y_true.float().cpu(), y_prot.float().cpu(), 'aod')
            obj = .75*abs(bias)+(1-.75)*(1-best_acc)
            return rocauc_score, acc, bias, obj
        return _get_results

    rocauc_score, acc, bias, obj = val_model(best_model, testloader, get_best_objective_results(best_thresh))

    print('roc auc', rocauc_score)
    print('accuracy with best thresh', acc)
    print('aod', bias.item())
    print('objective', obj.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for CelebA experiments')
    parser.add_argument('--trainsize', type=int, default=5000, help='Size of training set')
    parser.add_argument('--testsize', type=int, default=1000, help='Size of test set')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of worker threads')
    parser.add_argument('--print_priors', type=bool, default=True, help='Compute the prior percents')

    args = parser.parse_args()
    main(args)
