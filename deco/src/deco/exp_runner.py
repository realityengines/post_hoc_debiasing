import argparse

import yaml
from skopt import gbrt_minimize, gp_minimize, dummy_minimize,  forest_minimize
import os
from FairnessExperiment import *

OPTIMIZERS = {
    "dummy_minimize": dummy_minimize,
    "gp_minimize": gp_minimize,
    "forest_minimize": forest_minimize,
    "gbrt_minimize": gbrt_minimize
}


def main(layer, experiment_directory, trial_number, seed, model_num):
    """
    data_files:
        train: "/home/author/fairness/data/AdultDataset_train_1.csv"
        val: "/home/author/fairness/data/AdultDataset_val_1.csv"
        test: "/home/author/fairness/data/AdultDataset_test_1.csv"

    fairness:
        protected_feature: sex
        privileged_value: 1
        output_label: income-per-year

    model:
        num_hidden_layers: 4
        input_layer_width: 12,
        hidden_layers_width: 12,


    prediction:
        threshold = 0.5,

    optimization:
        metrics:
            - statistical_parity_difference
        optimizer:  gp_minimize
        layers: [1, 2, 3, 4, 5, 6]

    """

    np.random.seed(seed)
    with open(os.path.join(experiment_directory, 'config.yaml')) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    train_df = pd.read_csv(config["data_files"]["train"], error_bad_lines=False)
    val_df = pd.read_csv(config["data_files"]["val"], error_bad_lines=False)
    test_df = pd.read_csv(config["data_files"]["test"], error_bad_lines=False)

    fairness_experiment = FairnessExperiment(
        train_df,
        val_df,
        test_df,
        protected_feature=config["fairness"]["protected_feature"],
        privileged_value=config["fairness"]["privileged_value"],
        output_label=config["fairness"]["output_label"],
        num_hidden_layers=config["model"]["num_hidden_layers"],
        input_layer_width=config["model"]["input_layer_width"],
        hidden_layers_width=config["model"]["hidden_layers_width"],
        threshold=config["prediction"]["threshold"],
        metrics=config["optimization"]["metrics"],
        optimizer=OPTIMIZERS[config["optimization"]["optimizer"]])

    trial_dir = os.path.join(experiment_directory, "trial_" + str(trial_number))

    fairness_experiment.load_model(os.path.join("/home/author/fairness/deco/experiments", f"model_info_{model_num}.p"))

    #fairness_experiment.load_model("/home/author/fairness/deco/experiments/exp/statistical_parity_difference/run_0/model_info.p")
    #fairness_experiment.load_model(os.path.join(experiment_directory, "model_info.p"))

    os.system(f"echo {str(time.time())}> {os.path.join(trial_dir, f'layer_{layer}_optimization_start_time')}")
    res, func, model_info, history = fairness_experiment.run_on_layer(layer, config["optimization"]["acq_func"],
                                                                      config["optimization"]["n_calls"], 0.70, 20, 1)
    os.system(f"echo {str(time.time())}> {os.path.join(trial_dir, f'layer_{layer}_optimization_stop_time')}")

    sequence = []
    for i in range(len(history)):
        hist = history[i]
        # weight = res.x_iters[i]
        sequence.append({'val_bias': hist['val_bias'],
                         'val_accuracy': hist['val_accuracy'],
                         'test_bias': hist['test_bias'],
                         'test_accuracy': hist['test_accuracy'],
                         "query": i})

    sequence_df = pd.DataFrame(sequence)
    sequence_df.to_csv(os.path.join(trial_dir, "history_" + str(layer) + ".csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("layer", help="layer to optimize",
                        type=int)
    parser.add_argument("dir", help="directory containing an experiment config.yaml file")
    parser.add_argument("trial_number", help="trial number",
                        type=int)
    parser.add_argument("seed", help="random seed",
                        type=int)
    parser.add_argument("model_num", help="model number",
                        type=int)

    args = parser.parse_args()

    main(args.layer, args.dir, args.trial_number, args.seed, args.model_num)
