import argparse
import os
import random
from pathlib import Path

import yaml

# dropout=.5, width=32, height=10
# droput=.2, width=128, height=10
# dropout=.2, width=32, height=20
# dropout=.2, width=32, height=2


def main(args):
    Path(f'{args.dataset}_{args.metric}_{args.protected}').mkdir(exist_ok=True)
    args.num_runs = int(args.num_runs)
    args.protected = int(args.protected)
    args.architecture_type = int(args.architecture_type)
    hyperparam_dict = {
        'num_deep': [2, 5, 10, 15],
        'hid': [16, 32, 64, 128],
        'dropout_p': [0.1, 0.2, 0.3, 0.5]
    }

    if args.architecture_type == -1:
        hyperparam = {'num_deep': 10, 'hid': 32, 'dropout_p': 0.2}
    elif args.architecture_type == 0:
        hyperparam = {k: random.choice(v) for k, v in hyperparam_dict.items()}
    elif args.architecture_type == 1:
        hyperparam = {'num_deep': 10, 'hid': 32, 'dropout_p': 0.5}
    elif args.architecture_type == 2:
        hyperparam = {'num_deep': 10, 'hid': 64, 'dropout_p': 0.2}
    elif args.architecture_type == 3:
        hyperparam = {'num_deep': 10, 'hid': 16, 'dropout_p': 0.2}
    elif args.architecture_type == 4:
        hyperparam = {'num_deep': 20, 'hid': 32, 'dropout_p': 0.2}
    elif args.architecture_type == 5:
        hyperparam = {'num_deep': 2, 'hid': 32, 'dropout_p': 0.2}

    for i in range(args.num_runs):
        baselines_config = {
            'experiment_name': f'{args.dataset}_{args.metric}_{args.protected}_{i}_baselines',
            'dataset': args.dataset,
            'protected': args.protected,
            'modelpath': f'models/{args.dataset}_{args.architecture_type}_{i}_model.pt',
            'metric': args.metric,
            'models': [
                'default',
                'ROC',
                'EqOdds',
                'CalibEqOdds',
                'random',
                'adversarial'
            ],
            'CalibEqOdds': {'cost_constraint': 'fpr'},
            'random': {'num_trials': 201},
            'adversarial': {'epochs': 16, 'critic_steps': 201, 'actor_steps': 101, 'batch_size': 64, 'lambda': 0.75},
            'hyperparameters': hyperparam
        }

        with open(f'{args.dataset}_{args.metric}_{args.protected}/config_{args.dataset}_{args.metric}_{args.protected}_{i}_baselines.yaml', 'w') as fh:
            yaml.dump(baselines_config, fh)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", help="Which dataset")
    parser.add_argument("metric", help="which metric")
    parser.add_argument("protected", help="which protected")
    parser.add_argument("num_runs", help="Number of runs")
    parser.add_argument("architecture_type", help="Type of Architecture")

    args = parser.parse_args()

    main(args)
