import numpy as np
import pandas as pd

np.random.seed(0)
from aif360.metrics import ClassificationMetric
from aif360.datasets import *
from sklearn.metrics import accuracy_score
pd.options.mode.chained_assignment = None  # default='warn'


def classifier_score(pred_fun, metrics, df, protected_label, privileged_value, output_name, threshold, output_index):
    privilged_population = df[df[protected_label] == privileged_value]
    unprivilged_population = df[df[protected_label] != privileged_value]

    if (output_index):
        orig_privileged_scores = pred_fun(privilged_population.drop(columns=[output_name]).values)[:, output_index]
        orig_unprivileged_scores = pred_fun(unprivilged_population.drop(columns=[output_name]).values)[:, output_index]
    else:
        orig_privileged_scores = pred_fun(privilged_population.drop(columns=[output_name]).values)[:, 0]
        orig_unprivileged_scores = pred_fun(unprivilged_population.drop(columns=[output_name]).values)[:, 0]

    orig_scores = np.vstack([orig_privileged_scores.reshape(-1, 1),
                             orig_unprivileged_scores.reshape(-1, 1)])

    orig_scores[orig_scores <= 0.5] = 0
    orig_scores[orig_scores > 0.5] = 1

    original_privileged_accuracy = accuracy_from_scores(privilged_population[output_name], orig_privileged_scores,
                                                        threshold)
    original_unprivileged_accuracy = accuracy_from_scores(unprivilged_population[output_name], orig_unprivileged_scores,
                                                          threshold)

    original_accuracy = (len(privilged_population) * original_privileged_accuracy + len(
        unprivilged_population) * original_unprivileged_accuracy) / len(df)

    pre_transform_df = pd.concat([privilged_population, unprivilged_population])
    pre_transform_df[output_name] = np.hstack([orig_privileged_scores, orig_unprivileged_scores])
    pre_transform_df[output_name] = clip_at_threshold(pre_transform_df[output_name], 0.5)

    orig_dataset = BinaryLabelDataset(df=pd.concat([privilged_population, unprivilged_population]),
                                      favorable_label=1.0,
                                      unfavorable_label=0.0,
                                      label_names=[output_name],
                                      protected_attribute_names=[protected_label],
                                      privileged_protected_attributes=[privileged_value])

    pre_trans_dataset = BinaryLabelDataset(df=pre_transform_df,
                                           favorable_label=1.0,
                                           unfavorable_label=0.0,
                                           label_names=[output_name],
                                           protected_attribute_names=[protected_label],
                                           privileged_protected_attributes=[privileged_value])

    privileged_groups = {}
    privileged_groups[protected_label] = privileged_value
    unprivileged_groups = {}
    unprivileged_groups[protected_label] = 1 - privileged_value

    pre_trans_dataset.scores = pre_transform_df[output_name]
    dataset = ClassificationMetric(orig_dataset,
                                   pre_trans_dataset,
                                   privileged_groups=[privileged_groups],
                                   unprivileged_groups=[unprivileged_groups])
    return np.asanyarray([getattr(dataset, metric)() for metric in metrics])


def clip_at_threshold(y, threshold=0.5):
    y[y >= threshold] = 1
    y[y < threshold] = 0
    return y


def accuracy_from_scores(y_true, y_score, threshold):
    y_pred = np.copy(y_score)
    y_pred[y_pred <= threshold] = 0
    y_pred[y_pred > threshold] = 1
    return accuracy_score(y_true, y_pred)
