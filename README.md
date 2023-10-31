# Protein flower

## Installation

```bash
# Conda environment with dependencies.
conda env update -f fm.yml

# Manually need to install torch-scatter.
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# Activate environment
conda activate fm

# Install local package.
# Current directory should be flow-matching/
pip install -e .
```

## Wandb

Our training relies on logging with wandb. Log in to Wandb and make an account.
Authorize Wandb [here](https://wandb.ai/authorize).

## Data

Download preprocessed SCOPe dataset (~280MB) hosted on dropbox: [link](https://www.dropbox.com/scl/fi/b8l0bqowi96hl21ycsmht/preprocessed_scope.tar.gz?rlkey=0h7uulr7ioyvzlap6a0rwpx0n&dl=0)

```bash
# Expand tar file.
tar -xvzf preprocessed_scope.tar.gz
rm preprocessed_scope.tar.gz
```
Your directory should now look like this 
```
├── analysis
├── configs
├── data
├── experiments
├── models
├── notebooks
├── openfold
│   ├── data
│   │   └── tools
│   ├── model
│   ├── np
│   │   └── relax
│   ├── resources
│   └── utils
└── preprocessed
```

## Run training


```bash
# Train on SCOPe up to length 128
python -W ignore experiments/train_se3_flows.py
```

## Run inference

```bash
# Sample 10 sequences per length between 60 and 128
python -W ignore experiments/inference_se3_flows.py
```

## To-do

- [x] Train on SwissProt
- [x] Add DDP inference
- [ ] Add SDE interpolant
- [ ] Add translation schedule variant
- [ ] Implement inpainting training
- [ ] Implement inpainting inference/benchmark

# Contributing
This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the Microsoft Open Source Code of Conduct. For more information see the Code of Conduct FAQ or contact opencode@microsoft.com with any additional questions or comments.

# Trademarks
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.