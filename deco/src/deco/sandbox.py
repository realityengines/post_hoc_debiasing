from FairnessExperiment import *
import pandas as pd
import sys

sys.path.insert(0, '/Users/author/repos/fairness/deco/src/deco')
from skopt import gbrt_minimize
from skopt import gp_minimize
from skopt import dummy_minimize

train_df = pd.read_csv("/home/author/fairness/data/AdultDataset_train_1.csv")
val_df = pd.read_csv("/home/author/fairness/data/AdultDataset_val_1.csv")
test_df = pd.read_csv("/home/author/fairness/data/AdultDataset_test_1.csv")


# optimizer param is either "nevergrad", "hyperopt" or an optimizer from sk-opt such as
# gp_minimize/gp_minimize/dummy_minimize

fairness_experiment = FairnessExperiment(
    train_df,
    val_df,
    test_df,
    protected_feature="race",
    privileged_value=1.0,
    output_label="income-per-year",
    num_hidden_layers=8,
    input_layer_width=12,
    hidden_layers_width=3,
    threshold=0.5,
    metrics=["statistical_parity_difference"],
    optimizer=gp_minimize)

result = fairness_experiment.run(100, 0.6, 12, True)
