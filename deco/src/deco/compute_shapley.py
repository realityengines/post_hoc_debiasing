import numpy as np
import pandas as pd
import pickle
import time
import argparse
import yaml
import keras
from utils import accuracy_from_scores
import fairness
import gc
from tqdm import tqdm
import os.path

# References
# [1]  https://christophm.github.io/interpretable-ml-book/shapley.html#the-shapley-value-in-detail

class Shapley(object):

    def __init__(self, nn, config):

        self.nn = nn
        self.config = config
        self.val_df = pd.read_csv(config["data_files"]["val"])
        self.layer_weights = self.nn.get_weights()

    def compute(self):
        return [self.__single_layer_contribution__(i) for i in tqdm(range(1, len(self.nn.layers)))]

    def compute_accuracy(self):
        X = self.val_df.drop(columns=[self.config["fairness"]["output_label"]]).values
        y = self.val_df[self.config["fairness"]["output_label"]].values

        y_preds = self.nn.predict(X)[:, 0]
        y_preds_orig = y_preds
        y_preds_orig[y_preds_orig <= self.config["prediction"]["threshold"]] = 0
        y_preds_orig[y_preds_orig > self.config["prediction"]["threshold"]] = 1
        return accuracy_from_scores(y, y_preds, self.config["prediction"]["threshold"])

    def payoff(self, model, weights):

        model.set_weights(weights)
        scores = fairness.classifier_score(pred_fun=model.predict,
                                           metrics=self.config["optimization"]["metrics"],
                                           df=self.val_df,
                                           protected_label=self.config["fairness"]["protected_feature"],
                                           privileged_value=self.config["fairness"]["privileged_value"],
                                           output_name=self.config["fairness"]["output_label"],
                                           threshold=0.5,
                                           output_index=0)

        scores = scores[np.isfinite(scores)]
        val_bias = np.max(np.abs(scores))

        val_accuracy = self.compute_accuracy()
        return (1 + val_bias) * (1 - val_accuracy)

    def __get_weights__(self, index, randomized_indices):

        if index in randomized_indices:
            return np.random.randn(*self.layer_weights[index].shape)
        else:
            return self.layer_weights[index]

    def __single_layer_contribution__(self, k):
        # This is the number of iterations where we calculate the difference in contribution.
        # This is the M variable in [1]
        total_iterations = 10

        total_number_of_layers = len(self.nn.layers)
        sum_of_contributions = 0.0

        model = keras.models.clone_model(self.nn)
        model.set_weights(self.nn.get_weights())

        for m in range(0, total_iterations):
            gc.collect()

            r = np.random.choice(total_number_of_layers)

            randomized_indices_all = np.hstack([np.random.choice(np.arange(total_number_of_layers), r), [k]])

            random_weights = [self.__get_weights__(i, randomized_indices_all) for i in range(total_number_of_layers)]

            w_plus = [random_weights[i] for i in range(k)] + \
                     [self.layer_weights[k]] + \
                     [random_weights[i] for i in range(k + 1, total_number_of_layers)]

            w_minus = random_weights

            v_with = self.payoff(model, w_plus)
            v_without = self.payoff(model, w_minus)

            sum_of_contributions = sum_of_contributions + (v_with - v_without)
            w_plus.clear()
            w_minus.clear()
            del w_plus[:]
            del w_minus[:]

        return sum_of_contributions / total_iterations


def main(input_file, output_file):
    with open(os.path.join(experiment_directory, 'config.yaml')) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    np.random.seed(round(time.time()))

    model_info = pickle.load(open(input_file, "rb"))
    nn = model_info['model']

    shapley = Shapley(nn, config)
    vals = shapley.compute()

    if (os.path.isfile(output_file)):
        print(".... adding to old values ...")
        df = pd.DataFrame([vals], columns=list(map(str, range(len(vals)))))
        df = pd.concat([df, pd.read_csv(output_file)])
    else:
        df = pd.DataFrame([vals])

    df.to_csv(output_file, index=False, header=list(map(str, range(len(vals)))))

    print(vals)
    print(np.argsort(np.abs(vals)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="directory containing an experiment config.yaml file")
    parser.add_argument("train_df_file")
    parser.add_argument("val_df_file")

    args = parser.parse_args()
    experiment_directory = args.dir

    main(input_file=os.path.join(experiment_directory, "model_info.p"),
         output_file=os.path.join(experiment_directory, "shapley.csv"),
         )
