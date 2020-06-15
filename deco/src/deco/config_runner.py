##########################################################################
#          RUN EXPERIMENTS BASED ON A CONFIG.YAML FILE
##########################################################################
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

import argparse
import os
import yaml
from skopt import gbrt_minimize
from skopt import gp_minimize
from skopt import dummy_minimize


from FairnessExperiment import *


def build_initial_model(config, experiment_directory):
    train_df = pd.read_csv(config["data_files"]["train"])
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
        optimizer=gbrt_minimize)

    #fairness_experiment.load_model(os.path.join("/home/author/fairness/deco/experiments", "model_info.p"))
    #fairness_experiment.save_model(os.path.join(experiment_directory, "model_info.p"))


def run_shapley(config, experiment_directory):
    os.system(f"echo {str(time.time())}> {os.path.join(experiment_directory, 'shapley_start_time')}")
    for i in range(config["shapley_iterations"]):
        os.system(
            f"python ./src/deco/compute_shapley.py {experiment_directory} {config['data_files']['train']} {config['data_files']['val']}")
    os.system(f"echo {str(time.time())}> {os.path.join(experiment_directory, 'shapley_stop_time')}")


def get_best_shapley_layers(run_dir):
    shapley_df = pd.read_csv(os.path.join(run_dir, "shapley.csv"))
    means = shapley_df.mean()
    min_layer = int(means.idxmin())
    max_layer = int(means.idxmax())
    return [max_layer, min_layer]


def run_configuration(experiment_directory, model_num):
    with open(os.path.join(experiment_directory, 'config.yaml')) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    top_level_seeds = config["seeds"]
    number_of_trials = config["optimization"]["number_of_trials"]
    run_number = 0
    # Each config has a top level run where one model is built.
    # For each run (therefore, each model), we have many optimizations trails
    for seed in top_level_seeds:
        run_dir = os.path.join(experiment_directory, "run_" + str(run_number))
        run_number = run_number + 1

        if not os.path.isdir(run_dir):
            os.mkdir(run_dir)
            np.random.seed(seed)

            run_config = config.copy()
            del run_config["seeds"]

            copy_config_to_dir(run_config, run_dir)

            #build_initial_model(config, run_dir)
            #run_shapley(config, run_dir)

            if (config["select"] == 'shapley'):
                layers = get_best_shapley_layers(run_dir)
            else:
                layers = config['optimization']['layers']
            for trial in range(number_of_trials):
                trial_dir = os.path.join(run_dir, "trial_" + str(trial))
                trial_seed = trial
                if not os.path.isdir(trial_dir):

                    os.mkdir(trial_dir)
                    for layer in layers:
                        output_file = os.path.join(trial_dir, "out_" + str(layer))
                        print(os.system(
                            f'python -u ./src/deco/exp_runner.py {layer} {run_dir} {trial} {trial_seed} {model_num} > {output_file} '))
                else:
                    print(f"!!!!!!!!!!!! {trial_dir} EXISTS. PLEASE CLEAN UP !!!!!!!!!!!!.")
                    exit()
        else:
            print(f"!!!!!!!!!!!! {run_dir} EXISTS. PLEASE CLEAN UP !!!!!!!!!!!!.")
            exit()


def copy_config_to_dir(config, directory):
    with open(os.path.join(directory, 'config.yaml'), 'w') as config_file:
        yaml.dump(config, config_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="directory containing an experiment config.yaml file")
    parser.add_argument("model_num", help="model number")

    args = parser.parse_args()
    top_level_experiment_directory = args.dir
    model_num = args.model_num
    with open(os.path.join(top_level_experiment_directory, 'config.yaml')) as file:
        top_level_configuration = yaml.load(file, Loader=yaml.FullLoader)
        print("RUNNING WITH THIS CONFIG:")
        print(top_level_configuration)

        metrics = top_level_configuration["optimization"]["metrics"]

        for metric in metrics:
            metric_dir = os.path.join(top_level_experiment_directory, metric)
            if not os.path.isdir(metric_dir):
                os.mkdir(metric_dir)
                metric_config = top_level_configuration.copy()
                metric_config["optimization"]["metrics"] = [metric]

                copy_config_to_dir(metric_config, metric_dir)

                run_configuration(experiment_directory=metric_dir, model_num=model_num)

            else:
                print(f"!!!!!!!!!!!! {metric_dir} EXISTS. PLEASE CLEAN UP !!!!!!!!!!!!.")
                exit()

    # copy_command = f'gsutil -m rsync -r ../experiments/ gs://nsg-fairness-experiments/'
    # os.system(copy_command)
