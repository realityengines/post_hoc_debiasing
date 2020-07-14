"""
post_hoc_lib.py

Library for users to debias their own models
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


def val_model(model, loader, criterion, protected_index, prediction_index, lam=0.75):
    """Validate model on loader with criterion function"""
    y_true, y_pred, y_prot = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels, protected = inputs.to(device), labels[:, prediction_index].float().to(device), labels[:, protected_index].float().to(device)
            y_true.append(labels)
            y_prot.append(protected)
            y_pred.append(torch.sigmoid(model(inputs)[:, 0]))
    y_true, y_pred, y_prot = torch.cat(y_true), torch.cat(y_pred), torch.cat(y_prot)
    return criterion(y_true, y_pred, y_prot, lam)


def get_best_accuracy(y_true, y_pred, *_):
    """Select threshold that maximizes accuracy"""
    threshs = torch.linspace(0, 1, 1001)
    best_acc, best_thresh = 0., 0.
    for thresh in threshs:
        acc = torch.mean(((y_pred > thresh) == y_true).float()).item()
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh
    return best_acc, best_thresh


def compute_bias(y_pred, y_true, priv, metric):
    """Compute bias on the dataset"""
    def zero_if_nan(data):
        """Zero if there is a nan"""
        return 0. if torch.isnan(data) else data

    gtpr_priv = zero_if_nan(y_pred[priv * y_true == 1].mean())
    gfpr_priv = zero_if_nan(y_pred[priv * (1-y_true) == 1].mean())
    mean_priv = zero_if_nan(y_pred[priv == 1].mean())

    gtpr_unpriv = zero_if_nan(y_pred[(1-priv) * y_true == 1].mean())
    gfpr_unpriv = zero_if_nan(y_pred[(1-priv) * (1-y_true) == 1].mean())
    mean_unpriv = zero_if_nan(y_pred[(1-priv) == 1].mean())

    if metric == "spd":
        return mean_unpriv - mean_priv
    if metric == "aod":
        return 0.5 * ((gfpr_unpriv - gfpr_priv) + (gtpr_unpriv - gtpr_priv))
    if metric == "eod":
        return gtpr_unpriv - gtpr_priv


def get_objective_results(best_thresh):
    """Get the objective results with the best_threshold"""
    def _get_results(y_true, y_pred, y_prot, lam):
        """Inner function to be returned"""
        rocauc_score = roc_auc_score(y_true.cpu(), y_pred.cpu())
        acc = torch.mean(((y_pred > best_thresh) == y_true).float()).item()
        bias = compute_bias((y_pred > best_thresh).float().cpu(), y_true.float().cpu(), 1-y_prot.float().cpu(), 'aod')
        obj = lam*abs(bias)+(1-lam)*(1-acc)
        return rocauc_score, acc, bias, obj
    return _get_results


class Critic(nn.Module):
    """Critic class for adversarial debiasing method"""

    def __init__(self, sizein, num_deep=3, hid=32):
        super().__init__()
        self.fc0 = nn.Linear(sizein, hid)
        self.fcs = nn.ModuleList([nn.Linear(hid, hid) for _ in range(num_deep)])
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(hid, 1)

    def forward(self, t):
        t = t.reshape(1, -1)
        t = self.fc0(t)
        for fully_connected in self.fcs:
            t = F.relu(fully_connected(t))
            t = self.dropout(t)
        return self.out(t)


def get_best_objective(y_true, y_pred, y_prot, lam):
    """Find the threshold for the best objective"""
    threshs = torch.linspace(0, 1, 501)
    best_obj, best_thresh = math.inf, 0.
    for thresh in threshs:
        acc = torch.mean(((y_pred > thresh) == y_true).float()).item()
        bias = compute_bias((y_pred > thresh).float().cpu(), y_true.float().cpu(), 1-y_prot.float().cpu(), 'aod')
        obj = lam*abs(bias)+(1-lam)*(1-acc)
        if obj < best_obj:
            best_obj, best_thresh = obj, thresh
    return best_obj, best_thresh


class DebiasModel(object):
    """
    Abstract Base Class for user to overwrite with custom methods
    """

    def __init__(self):
        self.best_rand_model, self.best_rand_thresh = None, 0.
        self.best_adv_model, self.best_adv_thresh = None, 0.
        self.lam = 0.75

    @property
    def protected_index(self):
        """index for protected attribute"""
        raise NotImplementedError()

    @property
    def prediction_index(self):
        """index for prediction attribute"""
        raise NotImplementedError()

    def get_valloader(self):
        """get the valloader"""
        raise NotImplementedError()

    def get_testloader(self):
        """get the testloader"""
        raise NotImplementedError()

    def get_model(self):
        """get model and load weights"""
        raise NotImplementedError()

    def get_last_layer_name(self):
        """get name of last fully connected layer of network."""
        raise NotImplementedError()

    def _evaluate_model_thresh(self, model, best_thresh, verbose=True):
        rocauc_score, best_acc, bias, obj = val_model(
            model,
            self.get_testloader(),
            get_objective_results(best_thresh),
            self.protected_index,
            self.prediction_index,
            self.lam
        )

        if verbose:
            print()
            print('-'*20)
            print('Model Results')
            print('='*20)
            print('roc auc', rocauc_score)
            print('accuracy with best thresh', best_acc)
            print('aod', bias.item())
            print('objective', obj.item())
            print('-'*20)
            print()

        return {
            'roc_auc': float(rocauc_score),
            'accuracy': float(best_acc),
            'bias': float(bias.item()),
            'objective': float(obj.item())
        }

    def evaluate_original(self, verbose=True):
        """Evaluate Original Model"""
        _, best_thresh = val_model(
            self.get_model(),
            self.get_valloader(),
            get_best_accuracy,
            self.protected_index,
            self.prediction_index,
            self.lam
        )
        return self._evaluate_model_thresh(self.get_model(), best_thresh, verbose)

    def random_debias_model(self, num_rounds=101, verbose=True):
        """
        Run the random debiasing post hoc technique
        """
        net = self.get_model()
        valloader = self.get_valloader()
        rand_result = [math.inf, None, -1]
        rand_model = copy.deepcopy(net)
        for iteration in range(num_rounds):
            rand_model.to(device)
            for param in rand_model.parameters():
                param.data = param.data * (torch.randn_like(param) * 0.1 + 1)

            rand_model.eval()
            best_obj, best_thresh = val_model(rand_model, valloader, get_best_objective, self.protected_index, self.prediction_index, self.lam)
            if best_obj < rand_result[0]:
                del rand_result[1]
                rand_result = [best_obj, rand_model.state_dict(), best_thresh]

            if iteration % 10 == 0 and verbose:
                print(f"{iteration} / 101 trials have been sampled.")

        # evaluate best random model
        best_model = copy.deepcopy(net)
        best_model.load_state_dict(rand_result[1])
        best_model.to(device)
        best_thresh = rand_result[2]

        self.best_rand_model, self.best_rand_thresh = best_model, best_thresh
        return self.best_rand_model, self.best_rand_thresh

    def evaluate_random_debiased(self, verbose=True):
        """Evaluate random debiased model"""
        return self._evaluate_model_thresh(self.best_rand_model, self.best_rand_thresh, verbose)

    def adversarial_debias_model(self, batch_size=32, actor_steps=100, critic_steps=300, epochs=10, lam=0.75):
        """
        Run the adversarial debiasing post hoc technique
        """
        net = self.get_model()
        valloader = self.get_valloader()
        base_model = copy.deepcopy(net)
        base_last_layer = base_model.__getattr__(self.get_last_layer_name())
        base_model.__setattr__(self.get_last_layer_name(), nn.Linear(base_last_layer.in_features, base_last_layer.in_features))

        actor = nn.Sequential(base_model, nn.Linear(base_last_layer.in_features, 2))
        actor.to(device)
        actor_optimizer = optim.Adam(actor.parameters())
        actor_loss_fn = nn.BCEWithLogitsLoss()
        actor_loss = 0.

        critic = Critic(batch_size*base_last_layer.in_features)
        critic.to(device)
        critic_optimizer = optim.Adam(critic.parameters())
        critic_loss_fn = nn.MSELoss()
        critic_loss = 0.

        for epoch in range(epochs):
            for param in critic.parameters():
                param.requires_grad = True
            for param in actor.parameters():
                param.requires_grad = False
            actor.eval()
            critic.train()
            for step, (inputs, labels) in enumerate(valloader):
                if step > critic_steps:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                if inputs.size(0) != batch_size:
                    continue
                critic_optimizer.zero_grad()

                with torch.no_grad():
                    y_pred = actor(inputs)

                y_true = labels[:, self.prediction_index].float().to(device)
                y_prot = labels[:, self.protected_index].float().to(device)

                bias = compute_bias(y_pred, y_true, 1-y_prot, 'aod')
                res = critic(base_model(inputs))
                loss = critic_loss_fn(bias.unsqueeze(0), res[0])
                loss.backward()
                critic_loss += loss.item()
                critic_optimizer.step()
                if step % 100 == 0:
                    print_loss = critic_loss if (epoch*critic_steps + step) == 0 else critic_loss / (epoch*critic_steps + step)
                    print(f'=======> Epoch: {(epoch, step)} Critic loss: {print_loss:.3f}')

            for param in critic.parameters():
                param.requires_grad = False
            for param in actor.parameters():
                param.requires_grad = True
            actor.train()
            critic.eval()
            for step, (inputs, labels) in enumerate(valloader):
                if step > actor_steps:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                if inputs.size(0) != batch_size:
                    continue
                actor_optimizer.zero_grad()

                y_true = labels[:, self.prediction_index].float().to(device)
                y_prot = labels[:, self.protected_index].float().to(device)

                est_bias = critic(base_model(inputs))
                loss = actor_loss_fn(actor(inputs)[:, 0], y_true)
                loss = lam*abs(est_bias) + (1-lam)*loss

                loss.backward()
                actor_loss += loss.item()
                actor_optimizer.step()
                if step % 100 == 0:
                    print_loss = critic_loss if (epoch*actor_steps + step) == 0 else critic_loss / (epoch*actor_steps + step)
                    print(f'=======> Epoch: {(epoch, step)} Actor loss: {print_loss:.3f}')

        _, best_thresh = val_model(actor, valloader, get_best_objective, self.protected_index, self.prediction_index, self.lam)

        self.best_adv_model, self.best_adv_thresh = actor, best_thresh
        return self.best_adv_model, self.best_adv_thresh

    def evaluate_adversarial_debiased(self, verbose=True):
        """Evaluate adversarial debiased model"""
        return self._evaluate_model_thresh(self.best_adv_model, self.best_adv_thresh, verbose)


def get_objective_with_best_accuracy(y_true, y_pred, y_prot, lam):
    """Get objective for best accuracy threshold"""
    rocauc_score = roc_auc_score(y_true.cpu(), y_pred.cpu())
    best_acc, best_thresh = get_best_accuracy(y_true, y_pred, y_prot, lam)
    bias = compute_bias((y_pred > best_thresh).float().cpu(), y_true.float().cpu(), 1-y_prot.float().cpu(), 'aod')
    obj = lam*abs(bias)+(1-lam)*(1-best_acc)
    return rocauc_score, best_acc, bias, obj
