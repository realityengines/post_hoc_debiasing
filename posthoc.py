import argparse
import copy
import gc
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from aif360.algorithms.postprocessing import (
    CalibratedEqOddsPostprocessing,
    EqOddsPostprocessing,
    RejectOptionClassification
)
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import (
    load_preproc_data_german
)
from aif360.datasets import AdultDataset, BankDataset, CompasDataset
from aif360.metrics import ClassificationMetric
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_data(dataset_used, protected_attribute_used):
    if dataset_used == "adult":
        dataset_orig = AdultDataset()
        if protected_attribute_used == 1:
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]

    elif dataset_used == "german":
        dataset_orig = load_preproc_data_german()
        dataset_orig.labels -= 1
        if protected_attribute_used == 1:
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]

    elif dataset_used == "compas":
        dataset_orig = CompasDataset()
        if protected_attribute_used == 1:
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]

    elif dataset_used == "bank":
        dataset_orig = BankDataset()
        if protected_attribute_used == 1:
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]

    else:
        raise ValueError(f"{dataset_used} is not an available dataset.")

    dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.6], shuffle=True, seed=101)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True, seed=101)

    return dataset_orig_train, dataset_orig_valid, dataset_orig_test, privileged_groups, unprivileged_groups


class Model(nn.Module):

    def __init__(self, input_size, num_deep=10, hid=32, dropout_p=0.2):
        super().__init__()
        self.fc0 = nn.Linear(input_size, hid)
        self.bn0 = nn.BatchNorm1d(hid)
        self.fcs = nn.ModuleList([nn.Linear(hid, hid) for _ in range(num_deep)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hid) for _ in range(num_deep)])
        self.out = nn.Linear(hid, 2)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, t):
        t = self.bn0(self.dropout(F.relu(self.fc0(t))))
        for bn, fc in zip(self.bns, self.fcs):
            t = bn(self.dropout(F.relu(fc(t))))
        return torch.sigmoid(self.out(t))

    def trunc_forward(self, t):
        t = self.bn0(self.dropout(F.relu(self.fc0(t))))
        for bn, fc in zip(self.bns, self.fcs):
            t = bn(self.dropout(F.relu(fc(t))))
        return t


def load_model(input_size, config):
    if 'hyperparameters' in config:
        return Model(input_size, **config['hyperparameters'])
    else:
        return Model(input_size)


def train_model(model, X_train, y_train, X_valid, y_valid):
    loss_fn = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    patience = (math.inf, None, 0)
    patience_limit = 10
    for epoch in range(1001):
        model.train()
        batch_idxs = torch.split(torch.randperm(X_train.size(0)), 64)
        train_loss = 0
        for batch in batch_idxs:
            X = X_train[batch, :]
            y = y_train[batch]
            optimizer.zero_grad()
            loss = loss_fn(model(X)[:, 0], y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = loss_fn(model(X_valid)[:, 0], y_valid)
        scheduler.step(valid_loss)
        if epoch % 10 == 0:
            if valid_loss > patience[0]:
                patience = (patience[0], patience[1], patience[2]+1)
            else:
                patience = (valid_loss, model.state_dict(), 0)
            if patience[2] > patience_limit:
                print("Ending early, patience limit has been crossed without an increase in validation loss!")
                model.load_state_dict(patience[1])
                break
            print(f'=======> Epoch: {epoch} Train loss: {train_loss / len(batch_idxs)} Valid loss: {valid_loss} Patience valid loss: {patience[0]}')


class Critic(nn.Module):

    def __init__(self, sizein, num_deep=3, hid=32):
        super().__init__()
        self.fc0 = nn.Linear(sizein, hid)
        self.fcs = nn.ModuleList([nn.Linear(hid, hid) for _ in range(num_deep)])
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(hid, 1)

    def forward(self, t):
        t = t.reshape(1, -1)
        t = self.fc0(t)
        for fc in self.fcs:
            t = F.relu(fc(t))
            t = self.dropout(t)
        return self.out(t)


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


def objective_function(bias, performance, lam=0.75):
    return lam*abs(bias) + (1-lam)*(1-performance)


def get_objective(y_pred, y_true, priv, metric):
    bias = compute_bias(y_pred, y_true, priv, metric)
    performance = accuracy_score(y_true, y_pred)
    objective = objective_function(bias, performance)
    return {'objective': objective, 'bias': bias, 'performance': performance}


def main(config):
    # Setup directories to save models and results
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)

    # Get Data
    logger.info(f"Loading Data from dataset: {config['dataset']}.")
    train, valid, test, priv, unpriv = get_data(config['dataset'], config['protected'])
    priv_index = train.protected_attribute_names.index(list(priv[0].keys())[0])

    scale_orig = StandardScaler()
    X_train = torch.tensor(scale_orig.fit_transform(train.features), dtype=torch.float32)
    y_train = torch.tensor(train.labels.ravel(), dtype=torch.float32)
    # p_train = train.protected_attributes[:, priv_index]

    X_valid = torch.tensor(scale_orig.transform(valid.features), dtype=torch.float32)
    y_valid = torch.tensor(valid.labels.ravel(), dtype=torch.float32)
    p_valid = valid.protected_attributes[:, priv_index]

    X_test = torch.tensor(scale_orig.transform(test.features), dtype=torch.float32)
    y_test = torch.tensor(test.labels.ravel(), dtype=torch.float32)
    p_test = test.protected_attributes[:, priv_index]

    # Get Pretrained Model
    model = load_model(X_train.size(1), config)
    if Path(config['modelpath']).is_file():
        logger.info(f"Loading Model from {config['modelpath']}.")
        model.load_state_dict(torch.load(config['modelpath']))
    else:
        logger.info(f"{config['modelpath']} does not exist. Retraining model from scratch.")
        train_model(model, X_train, y_train, X_valid, y_valid)
        torch.save(model.state_dict(), config['modelpath'])
    model_state_dict = copy.deepcopy(model.state_dict())
    train = None

    # Preliminaries
    logger.info("Setting up preliminaries.")
    model.eval()
    with torch.no_grad():
        # train_pred = train.copy(deepcopy=True)
        # train_pred.scores = model(X_train)[:, 0].reshape(-1, 1).numpy()

        valid_pred = valid.copy(deepcopy=True)
        valid_pred.scores = model(X_valid)[:, 0].reshape(-1, 1).numpy()

        test_pred = test.copy(deepcopy=True)
        test_pred.scores = model(X_test)[:, 0].reshape(-1, 1).numpy()

    def get_valid_objective(y_pred):
        return get_objective(y_pred, y_valid.numpy(), p_valid, config['metric'])

    def get_test_objective(y_pred):
        return get_objective(y_pred, y_test.numpy(), p_test, config['metric'])

    results_valid = {}
    results_test = {}

    # Evaluate default model
    if "default" in config['models']:
        logger.info("Finding best threshold for default model to minimize objective function")
        threshs = np.linspace(0, 1, 1001)
        accuracies = []
        for thresh in threshs:
            acc = accuracy_score(y_valid, valid_pred.scores > thresh)
            accuracies.append(acc)
        best_thresh = threshs[np.argmax(accuracies)]

        logger.info("Evaluating default model with best threshold.")
        model.eval()
        with torch.no_grad():
            y_pred = (model(X_valid)[:, 0] > best_thresh).reshape(-1).numpy()
        results_valid['default'] = get_valid_objective(y_pred)

        model.eval()
        with torch.no_grad():
            y_pred = (model(X_test)[:, 0] > best_thresh).reshape(-1).numpy()
        results_test['default'] = get_test_objective(y_pred)

    # Evaluate ROC
    if "ROC" in config['models']:
        metric_map = {'spd': "Statistical parity difference", 'aod': "Average odds difference", 'eod': "Equal opportunity difference"}
        ROC = RejectOptionClassification(unprivileged_groups=unpriv,
                                         privileged_groups=priv,
                                         low_class_thresh=0.01, high_class_thresh=0.99,
                                         num_class_thresh=100, num_ROC_margin=50,
                                         metric_name=metric_map[config['metric']],
                                         metric_ub=0.05, metric_lb=-0.05)

        logger.info("Training ROC model with validation dataset.")
        ROC = ROC.fit(valid, valid_pred)

        logger.info("Evaluating ROC model.")
        y_pred = ROC.predict(valid_pred).labels.reshape(-1)
        results_valid['ROC'] = get_valid_objective(y_pred)

        y_pred = ROC.predict(test_pred).labels.reshape(-1)
        results_test['ROC'] = get_test_objective(y_pred)
        ROC = None

    # Evaluate Equality of Odds
    if "EqOdds" in config['models']:
        eo = EqOddsPostprocessing(privileged_groups=priv,
                                  unprivileged_groups=unpriv)

        logger.info("Training Equality of Odds model with validation dataset.")
        eo = eo.fit(valid, valid_pred)

        logger.info("Evaluating Equality of Odds model.")
        y_pred = eo.predict(valid_pred).labels.reshape(-1)
        results_valid['EqOdds'] = get_valid_objective(y_pred)

        y_pred = eo.predict(test_pred).labels.reshape(-1)
        results_test['EqOdds'] = get_test_objective(y_pred)
        eo = None

    # Evaluate Calibrated Equality of Odds
    if "CalibEqOdds" in config['models']:
        cost_constraint = config['CalibEqOdds']['cost_constraint']

        cpp = CalibratedEqOddsPostprocessing(privileged_groups=priv,
                                             unprivileged_groups=unpriv,
                                             cost_constraint=cost_constraint)

        logger.info("Training Calibrated Equality of Odds model with validation dataset.")
        cpp = cpp.fit(valid, valid_pred)

        logger.info("Evaluating Calibrated Equality of Odds model.")
        y_pred = cpp.predict(valid_pred).labels.reshape(-1)
        results_valid['CalibEqOdds'] = get_valid_objective(y_pred)

        y_pred = cpp.predict(test_pred).labels.reshape(-1)
        results_test['CalibEqOdds'] = get_test_objective(y_pred)

        cpp = None

    # Evaluate Random Debiasing
    if "random" in config['models']:
        logger.info("Generating Random Debiased models.")
        rand_result = [math.inf, None, -1]
        rand_model = load_model(X_train.size(1), config)
        for iteration in range(config['random']['num_trials']):
            rand_model.load_state_dict(model_state_dict)
            for param in rand_model.parameters():
                param.data = param.data * (torch.randn_like(param) * 0.1 + 1)

            rand_model.eval()
            with torch.no_grad():
                scores = rand_model(X_valid)[:, 0].reshape(-1).numpy()

            threshs = np.linspace(0, 1, 501)
            objectives = []
            for thresh in threshs:
                objectives.append(get_valid_objective(scores > thresh)['objective'])
            best_rand_thresh = threshs[np.argmin(objectives)]
            best_obj = np.min(objectives)
            if best_obj < rand_result[0]:
                del rand_result[1]
                rand_result = [best_obj, rand_model.state_dict(), best_rand_thresh]
            gc.collect()

            if iteration % 10 == 0:
                logger.info(f"{iteration} / {config['random']['num_trials']} trials have been sampled.")

        logger.info("Evaluating best random debiased model.")
        rand_model.load_state_dict(rand_result[1])
        rand_model.eval()
        with torch.no_grad():
            y_pred = (rand_model(X_valid)[:, 0] > rand_result[2]).reshape(-1).numpy()
        results_valid['Random'] = get_valid_objective(y_pred)

        rand_model.eval()
        with torch.no_grad():
            y_pred = (rand_model(X_test)[:, 0] > rand_result[2]).reshape(-1).numpy()
        results_test['Random'] = get_test_objective(y_pred)

        objectives = None
        rand_model = None
        rand_result = None

    # Evaluate Adversarial
    if "adversarial" in config['models']:
        logger.info("Training Adversarial model.")
        actor = load_model(X_train.size(1), config)
        actor.load_state_dict(model_state_dict)
        critic = Critic(config.get('hyperparameters', {'hid': 32})['hid']*config['adversarial']['batch_size'])
        critic_optimizer = optim.Adam(critic.parameters())
        critic_loss_fn = torch.nn.MSELoss()

        actor_optimizer = optim.Adam(actor.parameters())
        actor_loss_fn = torch.nn.BCELoss()

        for epoch in range(config['adversarial']['epochs']):
            for param in critic.parameters():
                param.requires_grad = True
            for param in actor.parameters():
                param.requires_grad = False
            actor.eval()
            critic.train()
            for step in range(config['adversarial']['critic_steps']):
                critic_optimizer.zero_grad()
                indices = torch.randint(0, X_valid.size(0), (config['adversarial']['batch_size'],))
                cy_valid = y_valid[indices]
                cX_valid = X_valid[indices]
                cp_valid = p_valid[indices]
                with torch.no_grad():
                    scores = actor(cX_valid)[:, 0].reshape(-1).numpy()

                bias = compute_bias(scores, cy_valid.numpy(), cp_valid, config['metric'])

                res = critic(actor.trunc_forward(cX_valid))
                loss = critic_loss_fn(torch.tensor([bias]), res[0])
                loss.backward()
                train_loss = loss.item()
                critic_optimizer.step()
                if step % 100 == 0:
                    logger.info(f'=======> Epoch: {(epoch, step)} Critic loss: {train_loss}')

            for param in critic.parameters():
                param.requires_grad = False
            for param in actor.parameters():
                param.requires_grad = True
            actor.train()
            critic.eval()
            for step in range(config['adversarial']['actor_steps']):
                actor_optimizer.zero_grad()
                indices = torch.randint(0, X_valid.size(0), (config['adversarial']['batch_size'],))
                cy_valid = y_valid[indices]
                cX_valid = X_valid[indices]
                lam = config['adversarial']['lambda']

                bias = critic(actor.trunc_forward(cX_valid))
                loss = actor_loss_fn(actor(cX_valid)[:, 0], cy_valid)
                loss = lam*abs(bias) + (1-lam)*loss

                loss.backward()
                train_loss = loss.item()
                actor_optimizer.step()
                if step % 100 == 0:
                    logger.info(f'=======> Epoch: {(epoch, step)} Actor loss: {train_loss}')

        logger.info("Finding optimal threshold for Adversarial model.")
        with torch.no_grad():
            adv_pred = valid.copy(deepcopy=True)
            adv_pred.scores = actor(X_valid)[:, 0].reshape(-1, 1).numpy()

        threshs = np.linspace(0, 1, 1001)
        objectives = []
        for thresh in threshs:
            labels = adv_pred.scores > thresh
            results = get_valid_objective(labels)
            objectives.append(results['objective'])
        best_adv_thresh = threshs[np.argmin(objectives)]

        logger.info("Evaluating Adversarial model on best threshold.")
        with torch.no_grad():
            labels = (actor(X_valid)[:, 0] > best_adv_thresh).reshape(-1, 1).numpy()
        results_valid['adversarial'] = get_valid_objective(labels)

        with torch.no_grad():
            labels = (actor(X_test)[:, 0] > best_adv_thresh).reshape(-1, 1).numpy()
        results_test['adversarial'] = get_test_objective(labels)

    # Save Results
    logger.info(f"Validation Results: {results_valid}")
    logger.info(f"Saving validation results to {config['experiment_name']}_valid_output.json")
    with open(f"results/{config['experiment_name']}_valid_output.json", "w") as fh:
        json.dump(results_valid, fh)

    logger.info(f"Test Results: {results_test}")
    logger.info(f"Saving validation results to {config['experiment_name']}_test_output.json")
    with open(f"results/{config['experiment_name']}_test_output.json", "w") as fh:
        json.dump(results_test, fh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to configuration yaml file.")
    args = parser.parse_args()
    with open(args.config, 'r') as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)
    main(config)
