
import argparse
import os
import yaml
from skopt import gbrt_minimize
from skopt import gp_minimize
from skopt import dummy_minimize


from FairnessExperiment import *


train_df = pd.read_csv('/home/author/fairness/data/AdultDataset_train_1.csv')
val_df = pd.read_csv('/home/author/fairness/data/AdultDataset_val_1.csv', error_bad_lines=False)
test_df = pd.read_csv('/home/author/fairness/data/AdultDataset_test_1.csv')


for i in range(0,10):
    np.random.seed(i)

    fairness_experiment = FairnessExperiment(
        train_df,
        val_df,
        test_df,
        protected_feature="race",
        privileged_value=1,
        output_label="income-per-year",
        num_hidden_layers=10,
        input_layer_width=32,
        hidden_layers_width=32,
        threshold=0.5,
        metrics=["equal_opportunity_difference"],
        optimizer=gbrt_minimize)


    fairness_experiment.build_model()

    fairness_experiment.save_model(os.path.join("/home/author/fairness/deco/experiments", f"model_info_{i}.p"))
