# Post-Hoc Methods for Debiasing Neural Networks

[Post-Hoc Methods for Debiasing Neural Networks](https://arxiv.org/abs/2006.08564)\
Yash Savani, Colin White, Naveen Sundar Govindarajulu.\
_arXiv:2006.08564_.

## Three New Post-Hoc Techniques
In this work, we introduce three new fine-tuning techniques to reduce bias in pretrained neural networks: random perturbation, layer-wise optimization, and adversarial fine-tuning. All three techniques work for any group fairness constraint. We include code that compares our three proposed methods with three popular post-processing methods, across three datasets provided by [aif360](https://aif360.readthedocs.io/en/latest/modules/datasets.html), and three popular bias measures.

<p align="center">
<img src="analysis/images/debias_fig.png" alt="debias_fig" width="99%">
</p>

## Requirements
- pyyaml
- numpy
- torch
- aif360 == 0.3.0rc0
- sklearn
- numba
- jupyter

Install the requirements using 
```
$ pip install -r requirements.txt
```

## Run a Post-Hoc debiasing experiment

### Step 1 - Create Configs
Create a config yaml file required to run the experiment by running 

```
$ python create_configs.py <dataset> <bias measure> <protected variable> <number of replications>
```
For example:
```
$ python create_configs.py adult spd 1 10
```

where <dataset> is one of "adult" (ACI), "bank" (BM), or "compas" (COMPAS), <bias measure> is one of "spd" (statistical parity difference), "eod" (equal opportunity difference), or "aod" (average odds difference), <protected variable> is 1 or 2 (described below), and <number of replications> is the number of trials to run, which can be any positive integer. This will create a config directory `<dataset>_<bias measure>_<protected variable>` (for example `adult_spd_1`) including all the corresponding config files for the experiment.

A table describing the relationship between the protected variable index and dataset is given below.

| dataset   | 1   | 2    |
|:----------|:----|:-----|
| adult     | sex | race |
| compas    | sex | race |
| bank      | age | race |

To automatically create all 12 experiments used in the paper, run

```
$ bash create_all_configs.sh
```

### Step 2 - Run Experiments
Run all the experiments described by the config files in the config directory created in Step 1 by running 

```
$ python run_experiments.py <config directory>
```
For example
```
$ python run_experiments.py adult_spd_1/
```

This will run a `posthoc.py` experiment for each config file in the config directory. All the biased, pretrained neural network models are saved in the `models/` directory. All the results from the experiments are saved in the `results/` directory in JSON format.

The `posthoc.py` includes benchmark code for the 3 post-processing debiasing techniques provided by the [aif360](https://aif360.readthedocs.io/en/latest/modules/algorithms.html#module-aif360.algorithms.postprocessing) framework: reject option classification, equalized odds postprocessing, and calibrated equalized odds postprocessing. It also includes code for the random perturbation, and adversarial fine-tuning algorithms. To get results for the layer-wise optimization technique, follow the instructions in the deco directory.

### Step 3 - Analyze Results
To analyze the results of the experiments and get the plots shown below you can run through the `analyze_results.ipynb` jupyter notebook.


<p align="center">
  <img src="analysis/images/spd_results.png" alt="spd-results-debiasing" width="32%">
  <img src="analysis/images/pareto_plot_BM (age)_spd.png" alt="eod-results-debiasing" width="32%">
  <img src="analysis/images/multinet_bm_results.png" alt="multinet-results-debiasing" width="32%">
</p>

### Step 4 - Cleanup configs

To clean up the config directories run 
```
$ bash cleanup.sh
```
