# Project flower 💮

Also called SE(3) flow matching. All steps assume your current directory is 

```bash
feynman/projects/sampling/experimental/jyim/se3_flow_matching
```

## Installation

Set-up and activate the sampling environment.

```bash
conda env create -f infrastructure/sampling-environment.yml
conda activate sampling
```

We need additional packages. I might have missed some.

```bash
# Installs openfold as a package.
pip install -e .

# Third party packages.
pip install ml-collections
pip install dm-tree
pip install wandb

```

## Training

Baseline experiment. Flag `-W ignore` suppresses user warnings. Turn off if you want to see them or if you are debugging.

```bash
python -W ignore experiments/train_se3_flows.py
```

Overfitting a single example

```bash
python -W ignore experiments/train_se3_flows.py -cn overfit
```


To use Amulet, make sure you're in `feynman/projects/sampling`. Current experiment can be printed with `amlt list`. jobs can also be founded here https://ml.azure.com/runs.

Amulet CLI https://amulet-docs.azurewebsites.net/main/basics/45_monitoring.html 

Amulet configs https://amulet-docs.azurewebsites.net/main/config_file.html