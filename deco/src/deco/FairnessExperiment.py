import pickle
import time
import logging
from tqdm import tqdm
import pandas as pd
from aif360.algorithms.postprocessing import RejectOptionClassification, EqOddsPostprocessing, \
    CalibratedEqOddsPostprocessing
from aif360.datasets import *
from aif360.metrics import ClassificationMetric
from skopt import dummy_minimize

import benchmark
import fairness
from utils import *
from hyperopt import hp
from hyperopt import fmin, tpe, Trials, rand
import nevergrad as ng
import keras
logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None  # default='warn'

METRICS = ["statistical_parity_difference", "average_abs_odds_difference", "equal_opportunity_difference"]

class FairnessExperiment(object):

    def __init__(self,
                 train_df,
                 val_df,
                 test_df,
                 protected_feature,
                 privileged_value,
                 output_label,
                 num_hidden_layers,
                 input_layer_width,
                 hidden_layers_width,
                 threshold,
                 metrics,
                 optimizer):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.protected_feature = protected_feature
        self.privileged_value = privileged_value
        self.output_label = output_label

        self.num_hidden_layers = num_hidden_layers
        self.input_layer_width = input_layer_width
        self.hidden_layers_width = hidden_layers_width
        self.nn = None
        self.starting_model_info = None

        self.x_train = train_df.drop(columns=[output_label]).values
        self.x_val = val_df.drop(columns=[output_label]).values
        self.x_test = test_df.drop(columns=[output_label]).values

        self.y_train = train_df[output_label].values
        self.y_val = val_df[output_label].values
        self.y_test = test_df[output_label].values

        self.input_features_num = self.x_train.shape[1]

        self.threshold = threshold

        self.metrics = metrics

        self.history = []

        self.optimizer = optimizer
        self.baselines = None
        self.C = 0.25
        
    def load_model(self, path):

        starting_model_info = pickle.load(open(path, 'rb'))
        baselines = pickle.load(open(path + "_baselines.p", 'rb'))
        print(baselines)
        self.threshold = baselines["initial"]["threshold"]

        self.starting_model_info = starting_model_info
        self.nn = starting_model_info['model']
        
        
        scores = self.__classifier_score__(self.test_df, 0, self.nn)
        scores = scores[np.isfinite(scores)]
        total_bias = np.max(np.abs(scores))
        print("INITIAL MODEL BIAS:  " + str(total_bias))
        print("INITIAL MODEL ACCURACY:  " + str(self.starting_model_info['accuracy']))
        print("INITIAL MODEL OBJECTIVE: " + str(self.objective(total_bias, self.starting_model_info['accuracy'])))

    def get_all_baselines(self):
        baselines = {}

        start = time.time()
        baselines['reject_option_classification'] = self.baseline_reject_option_classification()
        end = time.time()
        baselines['reject_option_classification']["time"] = end - start
        print("--------")
        start = time.time()
        baselines['eq_odds_postprocessing'] = self.baseline_eq_odds_postprocessing()
        end = time.time()
        baselines['eq_odds_postprocessing']["time"] = end - start
        print("--------")

        start = time.time()
        baselines['calibrated eq_odds_postprocessing'] = self.baseline_calibrated_eq_odds_postprocessing()
        end = time.time()
        baselines['calibrated eq_odds_postprocessing']["time"] = end - start

        return baselines

    def save_model(self, path):
        print(str(self.baselines))
        pickle.dump(self.starting_model_info, open(path, "wb"))
        pickle.dump(self.baselines, open(path + "_baselines.p", "wb"))


    def build_model(self):
        self.starting_model_info = self.__train_and_measure_model__()

        self.starting_model_info['baselines'] = self.get_all_baselines()
        self.baselines = self.starting_model_info['baselines'] #self.get_all_baselines()
        self.baselines["initial"] = {
            "bias": self.starting_model_info['bias'],
            "accuracy": self.starting_model_info['accuracy'],
            "threshold": self.threshold
        }
        print("INITIAL MODEL BIAS:  " + str(self.starting_model_info['bias']))
        print("INITIAL MODEL ACCURACY:  " + str(self.starting_model_info['accuracy']))
        print("INITIAL MODEL OBJECTIVE: " + str(self.starting_model_info['objective']))

    def __train_and_measure_model__(self):
        nn = create_model_nn(self.num_hidden_layers,
                             self.input_layer_width,
                             self.hidden_layers_width,
                             self.input_features_num)
        self.nn = nn
        self.C = 0.25
        nn.fit(self.x_train, self.y_train, epochs=100, verbose=2, shuffle=True)
        accuracy = self.__get_accuracy__(self.x_test, self.y_test, self.nn, self.threshold)

        starting_model = keras.models.clone_model(self.nn)
        starting_model.set_weights(self.nn.get_weights())

        thresholds = np.linspace(0, 1, 101)
        accs = []
        for thresh in tqdm(thresholds):
            
            acc = self.__get_accuracy__(self.x_test, self.y_test, self.nn, thresh)
            accs.append(acc)

        best_accuracy = np.max(accs)
        best_threshold = thresholds[np.argmax(accs)]

        print(str(thresholds))
        print(str(accs))

        print(f"best threshold is {best_threshold}")

        self.threshold = best_threshold

        scores = self.__classifier_score__(self.test_df, 0, nn)
        scores = scores[np.isfinite(scores)]
        total_bias = np.max(np.abs(scores))

        return {"model": starting_model,
                "accuracy": best_accuracy,
                "threshold": best_threshold,
                "objective": self.objective(total_bias, accuracy),#
                "bias": total_bias}

    def objective(self, bias, accuracy):
        return (1 - self.C) * abs(bias) +  self.C * (1 - accuracy)

    def __get_accuracy__(self, X, y, nn, threshold):
        y_preds = nn.predict(X)[:, 0]
        return accuracy_from_scores(y, y_preds, threshold)

    ## Output index is to handle differences between the output matrix from sklearn and keras.
    ## The output index can be ignored if working in just keras.
    def __classifier_score__(self, scoring_df, output_index, nn):
        privileged_index = scoring_df[self.protected_feature] == self.privileged_value
        y_pred = nn.predict(scoring_df.drop(columns=[self.output_label]).values)
        y_true = scoring_df[self.output_label].values

        return compute_bias(y_pred, y_true, privileged_index,self.metrics[0])

    def __classifier_score__aif360_(self, scoring_df, output_index, nn):

        original_df = scoring_df
        predictions_df = original_df#.copy()
        predictions = nn.predict(original_df.drop(columns=[self.output_label]).values)
        predictions_df[self.output_label] = clip_at_threshold(predictions, self.threshold)

        orig_dataset = BinaryLabelDataset(df=original_df,
                                          favorable_label=1.0,
                                          unfavorable_label=0.0,
                                          label_names=[self.output_label],
                                          protected_attribute_names=[self.protected_feature],
                                          privileged_protected_attributes=[self.privileged_value])

        transformed_dataset = BinaryLabelDataset(df=predictions_df,
                                                 favorable_label=1.0,
                                                 unfavorable_label=0.0,
                                                 label_names=[self.output_label],
                                                 protected_attribute_names=[self.protected_feature],
                                                 privileged_protected_attributes=[self.privileged_value])

        privileged_groups = {}
        privileged_groups[self.protected_feature] = self.privileged_value
        unprivileged_groups = {}
        unprivileged_groups[self.protected_feature] = 1 - self.privileged_value

        transformed_dataset.scores = predictions
        classification_dataset = ClassificationMetric(orig_dataset,
                                                      transformed_dataset,
                                                      privileged_groups=[privileged_groups],
                                                      unprivileged_groups=[unprivileged_groups])
        return np.asanyarray([np.abs(getattr(classification_dataset, metric)()) for metric in self.metrics])

    def __get_bias_function_for__(self, layer, min_accuracy, output_index):
        nn = self.starting_model_info['model']
        # before_layers = [l.copy() for l in nn.layers[0:layer]]
        # before_nn = keras.Sequential(before_layers)
        # y_preds_val_before = before_nn.predict(self.x_val)[:, 0]

        # (1-lambda) * abs(bias) - lambda * accuracy
        def bias_function_nn(layer_weights, return_acc=False):
            layer_shape = nn.layers[layer].get_weights()[0].shape
            layer_bias = np.asanyarray(nn.layers[layer].get_weights()[1])
            nn.layers[layer].set_weights([np.asanyarray(layer_weights).reshape(*layer_shape),
                                          layer_bias])

            print("debug"
                  "")
            val_accuracy = self.__get_accuracy__(self.x_val, self.y_val, nn, self.threshold)
            val_scores = np.asanyarray(self.__classifier_score__(self.val_df, output_index, nn))
            val_scores = val_scores[np.isfinite(val_scores)]
            total_val_bias = np.abs(val_scores)[0]

            test_scores = np.asanyarray(self.__classifier_score__(self.test_df, output_index, nn))
            test_scores = test_scores[np.isfinite(test_scores)]
            total_test_bias = np.abs(test_scores)[0]
            test_accuracy = self.__get_accuracy__(self.x_test, self.y_test, nn, self.threshold)

            if return_acc:
                return self.objective(total_val_bias, val_accuracy), val_accuracy
            else:
                self.history = self.history + [{'val_bias': total_val_bias,
                                                'test_bias': total_test_bias,
                                                'test_accuracy': test_accuracy,
                                                'val_accuracy': val_accuracy}]

                print(f"Result ---> {self.objective(total_val_bias, val_accuracy)}")
                return self.objective(total_val_bias, val_accuracy)

            
        return bias_function_nn

    def run_on_layer(self, layer, acq_func, n_calls, min_accuracy, n_jobs, verbose):
        if not self.nn:
            self.build_model()

        fn = self.__get_bias_function_for__(layer, min_accuracy, 0)

        print(layer)
        starting_weights = self.nn.layers[layer].get_weights()[0]
        total_dimensions = starting_weights.size
        dimensions = [(-1.0, 1.0, starting_weights.reshape(-1)[i]) for i in range(total_dimensions)]
        space = [
            hp.uniform(str(i), -0.5 * abs(starting_weights.reshape(-1)[i]), abs(starting_weights.reshape(-1)[i]) * 0.5)
            for i in range(total_dimensions)]
        trials = Trials()

        # fn = lambda weights: np.abs(np.asanyarray(weights).mean())
        if self.optimizer == 'nevergrad':
            optimizer = ng.optimizers.TBPSA(parametrization=total_dimensions, budget=100)
            res = optimizer.minimize(fn, verbosity=2)  # best value

        elif self.optimizer == 'hyperopt':
            res = fmin(fn, space, algo=tpe.suggest, max_evals=n_calls, trials=trials)

        elif (self.optimizer == dummy_minimize):
            res = self.optimizer(fn,
                                 dimensions,
                                 n_calls=n_calls,
                                 verbose=verbose)

        else:
            res = self.optimizer(fn,
                                 dimensions,
                                 n_calls=100,
                                 acq_func=acq_func,
                                 verbose=verbose,
                                 n_jobs=n_jobs,
                                 n_points=100)

        return (res, trials), fn, self.starting_model_info, self.history

    def run(self, n_calls, min_accuracy, n_jobs, verbose):
        if not self.nn:
            self.build_model()

        return [self.run_on_layer(i, "LCB", n_calls, min_accuracy, n_jobs, verbose) for i in
                range(1, len(self.nn.layers))]

    METRICS = ["statistical_parity_difference", "average_abs_odds_difference", "equal_opportunity_difference"]

    def get_all_metrics(self, classification_metric):
        return {
            "statistical_parity_difference": np.abs(getattr(classification_metric, "statistical_parity_difference")()),
            "average_abs_odds_difference": np.abs(getattr(classification_metric, "average_abs_odds_difference")()),
            "equal_opportunity_difference": np.abs(getattr(classification_metric, "equal_opportunity_difference")())

        }

    def baseline_reject_option_classification(self):
        """
        Runs reject option classification.
        Needs a model to be loaded or trained.
        Trains reject option classification on the smaller validation set and tests on the test set.
        """
        dataset_orig_test, dataset_orig_test_pred, dataset_orig_valid, dataset_orig_valid_pred, metric_name, privileged_groups, unprivileged_groups = self.prepare_for_baseline_debiasing()

        ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups,
                                         low_class_thresh=0.01,
                                         high_class_thresh=0.99,
                                         num_class_thresh=100,
                                         num_ROC_margin=50,
                                         metric_name=metric_name,
                                         metric_ub=benchmark.metric_ub, metric_lb=benchmark.metric_lb)

        ROC = ROC.fit(dataset_orig_valid, dataset_orig_valid_pred)
        dataset_transf_valid_pred = ROC.predict(dataset_orig_valid_pred)
        dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)


        metric_valid_aft = benchmark.compute_metrics(dataset_orig_valid,
                                                    dataset_transf_valid_pred,
                                                    unprivileged_groups, privileged_groups)

        metric_test_aft = benchmark.compute_metrics(dataset_orig_test,
                                                    dataset_transf_test_pred,
                                                    unprivileged_groups, privileged_groups)

        return {'accuracy_valid': accuracy_from_scores(self.y_val, dataset_transf_valid_pred.scores, self.threshold),
                'accuracy_test': accuracy_from_scores(self.y_test, dataset_transf_test_pred.scores, self.threshold),
                'bias_valid': self.get_all_metrics(metric_valid_aft),
                'bias_test': self.get_all_metrics(metric_test_aft),
                }

    def baseline_eq_odds_postprocessing(self):
        dataset_orig_test, dataset_orig_test_pred, dataset_orig_valid, dataset_orig_valid_pred, metric_name, privileged_groups, unprivileged_groups = self.prepare_for_baseline_debiasing()

        EOP = EqOddsPostprocessing(unprivileged_groups=unprivileged_groups,
                                   privileged_groups=privileged_groups, )

        EOP = EOP.fit(dataset_orig_valid, dataset_orig_valid_pred)

        dataset_transf_valid_pred = EOP.predict(dataset_orig_valid_pred)
        dataset_transf_test_pred = EOP.predict(dataset_orig_test_pred)


        metric_valid_aft = benchmark.compute_metrics(dataset_orig_valid,
                                                    dataset_transf_valid_pred,
                                                    unprivileged_groups, privileged_groups)

        metric_test_aft = benchmark.compute_metrics(dataset_orig_test,
                                                    dataset_transf_test_pred,
                                                    unprivileged_groups, privileged_groups)

        return {'accuracy_valid': accuracy_from_scores(self.y_val, dataset_transf_valid_pred.scores, self.threshold),
                'accuracy_test': accuracy_from_scores(self.y_test, dataset_transf_test_pred.scores, self.threshold),
                'bias_valid': self.get_all_metrics(metric_valid_aft),
                'bias_test': self.get_all_metrics(metric_test_aft),
                }

    def baseline_calibrated_eq_odds_postprocessing(self):
        dataset_orig_test, dataset_orig_test_pred, dataset_orig_valid, dataset_orig_valid_pred, metric_name, privileged_groups, unprivileged_groups = self.prepare_for_baseline_debiasing()

        CEOP = CalibratedEqOddsPostprocessing(unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups, )

        CEOP = CEOP.fit(dataset_orig_valid, dataset_orig_valid_pred)

        dataset_transf_valid_pred = CEOP.predict(dataset_orig_valid_pred)
        dataset_transf_test_pred = CEOP.predict(dataset_orig_test_pred)

        metric_valid_aft = benchmark.compute_metrics(dataset_orig_valid,
                                                    dataset_transf_valid_pred,
                                                    unprivileged_groups, privileged_groups)

        metric_test_aft = benchmark.compute_metrics(dataset_orig_test,
                                                    dataset_transf_test_pred,
                                                    unprivileged_groups, privileged_groups)

        return {'accuracy_valid': accuracy_from_scores(self.y_val, dataset_transf_valid_pred.scores, self.threshold),
                'accuracy_test': accuracy_from_scores(self.y_test, dataset_transf_test_pred.scores, self.threshold),
                'bias_valid': self.get_all_metrics(metric_valid_aft),
                'bias_test': self.get_all_metrics(metric_test_aft),
                }

    def prepare_for_baseline_debiasing(self):
        dataset_orig_train = StandardDataset(df=self.train_df,
                                             label_name=self.output_label,
                                             favorable_classes=[self.privileged_value],
                                             protected_attribute_names=[self.protected_feature],
                                             privileged_classes=[[self.privileged_value]])
        dataset_orig_valid = StandardDataset(df=self.val_df,
                                             label_name=self.output_label,
                                             favorable_classes=[self.privileged_value],
                                             protected_attribute_names=[self.protected_feature],
                                             privileged_classes=[[self.privileged_value]])
        dataset_orig_test = StandardDataset(df=self.test_df,
                                            label_name=self.output_label,
                                            favorable_classes=[self.privileged_value],
                                            protected_attribute_names=[self.protected_feature],
                                            privileged_classes=[[self.privileged_value]])
        privileged_groups = [{self.protected_feature: self.privileged_value}]
        unprivileged_groups = [{self.protected_feature: 1 - self.privileged_value}]
        X_train = dataset_orig_train.features
        y_train_pred = self.nn.predict(X_train)
        # positive class index
        pos_ind = 1
        dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
        dataset_orig_train_pred.labels = y_train_pred
        dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
        X_valid = (dataset_orig_valid_pred.features)
        dataset_orig_valid_pred.scores = self.nn.predict(X_valid).reshape(-1, 1)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        X_test = dataset_orig_test_pred.features
        dataset_orig_test_pred.scores = self.nn.predict(X_test).reshape(-1, 1)
        metric_name = "Statistical parity difference"
        return dataset_orig_test, dataset_orig_test_pred, dataset_orig_valid, dataset_orig_valid_pred, metric_name, privileged_groups, unprivileged_groups
