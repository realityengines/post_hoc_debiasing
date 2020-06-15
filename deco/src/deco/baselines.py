from sklearn.metrics import accuracy_score
import numpy as np

from aif360.algorithms.postprocessing import RejectOptionClassification, EqOddsPostprocessing, \
    CalibratedEqOddsPostprocessing
from aif360.datasets import *
from aif360.metrics import ClassificationMetric

threshold = 0.5
#### upper and lower bound for metrics
metric_ub = 0.15
metric_lb = 0.0


def accuracy_from_scores(y_true, y_score, threshold):
    y_pred = np.copy(y_score)
    y_pred[y_pred <= threshold] = 0
    y_pred[y_pred > threshold] = 1
    return accuracy_score(y_true, y_pred)


def get_classification_metric_object(dataset_true,
                                     dataset_pred,
                                     unprivileged_groups,
                                     privileged_groups):
    """Build the ClassificationMetric object"""
    classified_metric_pred = ClassificationMetric(dataset_true,
                                                  dataset_pred,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)

    return classified_metric_pred


def baseline_reject_option_classification(nn, df,
                                          output_label, protected_feature, privileged_value,
                                          metric_name):
    """
    Runs reject option classification.
    Needs a model to be loaded or trained.
    Trains reject option classification on the smaller validation set and tests on the test set.
    """

    dataset_orig, dataset_pred, privileged_groups, unprivileged_groups = prepare_for_baseline_debiasing(nn, df,
                                                                                                        output_label,
                                                                                                        protected_feature,
                                                                                                        privileged_value)

    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups,
                                     low_class_thresh=0.01,
                                     high_class_thresh=0.99,
                                     num_class_thresh=100,
                                     num_ROC_margin=50,
                                     metric_name=metric_name,
                                     metric_ub=metric_ub,
                                     metric_lb=metric_lb)

    ROC = ROC.fit(dataset_orig, dataset_pred)
    dataset_transf_pred = ROC.predict(dataset_orig)

    metric_test_aft = get_classification_metric_object(dataset_orig,
                                                       dataset_transf_pred,
                                                       unprivileged_groups,
                                                       privileged_groups)

    y_val = df[output_label]
    return {'accuracy': accuracy_from_scores(y_val, dataset_transf_pred.scores, threshold),
            'bias': (getattr(metric_test_aft, metric_name))()}


def baseline_eq_odds_postprocessing(nn, df,
                                    output_label, protected_feature, privileged_value,
                                    metric_name):
    dataset_orig, dataset_pred, privileged_groups, unprivileged_groups = prepare_for_baseline_debiasing(nn, df,
                                                                                                        output_label,
                                                                                                        protected_feature,
                                                                                                        privileged_value)

    EOP = EqOddsPostprocessing(unprivileged_groups=unprivileged_groups,
                               privileged_groups=privileged_groups, )

    EOP = EOP.fit(dataset_orig, dataset_pred)
    dataset_transf_pred = EOP.predict(dataset_orig)

    metric_test_aft = get_classification_metric_object(dataset_orig,
                                                       dataset_transf_pred,
                                                       unprivileged_groups, privileged_groups)

    y_val = df[output_label]
    return {'accuracy': accuracy_from_scores(y_val, dataset_transf_pred.scores, threshold),
            'bias': (getattr(metric_test_aft, metric_name))()}


def baseline_calibrated_eq_odds_postprocessing(nn, df,
                                               output_label, protected_feature, privileged_value,
                                               metric_name):
    dataset_orig, dataset_pred, privileged_groups, unprivileged_groups = prepare_for_baseline_debiasing(nn, df,
                                                                                                        output_label,
                                                                                                        protected_feature,
                                                                                                        privileged_value)

    CEOP = CalibratedEqOddsPostprocessing(unprivileged_groups=unprivileged_groups,
                                          privileged_groups=privileged_groups, )

    CEOP = CEOP.fit(dataset_orig, dataset_pred)
    dataset_transf_pred = CEOP.predict(dataset_orig)

    metric_test_aft = get_classification_metric_object(dataset_orig,
                                                       dataset_transf_pred,
                                                       unprivileged_groups, privileged_groups)

    y_val = df[output_label]
    return {'accuracy': accuracy_from_scores(y_val, dataset_transf_pred.scores, threshold),
            'bias': (getattr(metric_test_aft, metric_name))()}


def prepare_for_baseline_debiasing(nn, df, output_label, protected_feature, privileged_value):
    dataset_orig = StandardDataset(df=df,
                                   label_name=output_label,
                                   favorable_classes=[privileged_value],
                                   protected_attribute_names=[protected_feature],
                                   privileged_classes=[[privileged_value]])

    privileged_groups = [{protected_feature: privileged_value}]
    unprivileged_groups = [{protected_feature: 1 - privileged_value}]

    dataset_pred = dataset_orig.copy(deepcopy=True)
    X = dataset_pred.features
    dataset_pred.scores = nn.predict(X).reshape(-1, 1)

    return dataset_orig, dataset_pred, \
           privileged_groups, unprivileged_groups
