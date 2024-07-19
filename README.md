# Fast protein backbone generation with SE(3) flow matching

> [!NOTE]  
> This code is the source code for https://arxiv.org/abs/2310.05297. 
> See the main branch for the improved FrameFlow trained on PDB and for motif-scaffolding.

If you use this work (or code) then please cite the paper.
```
@article{yim2023fast,
  title={Fast protein backbone generation with SE (3) flow matching},
  author={Yim, Jason and Campbell, Andrew and Foong, Andrew YK and Gastegger, Michael and Jim{\'e}nez-Luna, Jos{\'e} and Lewis, Sarah and Satorras, Victor Garcia and Veeling, Bastiaan S and Barzilay, Regina and Jaakkola, Tommi and others},
  journal={arXiv preprint arXiv:2310.05297},
  year={2023}
}
```

![frameflow-landing-page](https://github.com/microsoft/protein-frame-flow/blob/main/media/frame_flow_sampling.gif)


## Installation

```bash
# Conda environment with dependencies.
conda env create -f fm.yml

# Activate environment
conda activate fm

# Manually need to install torch-scatter.
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# Install local package.
# Current directory should be protein-frame-flow/
pip install -e .
```

## Wandb

Our training relies on logging with wandb. Log in to Wandb and make an account.
Authorize Wandb [here](https://wandb.ai/authorize).

## Data

Download preprocessed SCOPe dataset (~280MB) hosted on dropbox: [link](https://www.dropbox.com/scl/fi/b8l0bqowi96hl21ycsmht/preprocessed_scope.tar.gz?rlkey=0h7uulr7ioyvzlap6a0rwpx0n&dl=0).

Other datasets are also possible to train on using the `data/process_pdb_files.py` script.
However, we currently do not support other datasets.

```bash
# Expand tar file.
tar -xvzf preprocessed_scope.tar.gz
rm preprocessed_scope.tar.gz
```
Your directory should now look like this 
```
├── analysis
├── build
├── configs
├── data
├── experiments
├── media
├── models
├── openfold
├── preprocessed
└── weights
```

## Training

By default the code uses 2 GPUs with DDP and runs for 200 epochs.
We used 2 A6000 40GB GPUs on which training took ~2 days.
Following our paper, we train on SCOPe up to length 128.

```bash
python -W ignore experiments/train_se3_flows.py
```

## Inference

### Download weights

The published weights are hosted on dropbox: [link](https://www.dropbox.com/scl/fi/r8i0o057b0ms71ep5bf4m/published.ckpt?rlkey=pygthp5qjpwkn4glmai48mgy7&dl=0).
Download the checkpoint and place in the weights subdirectory.

```
weights
├── config.yaml
└── published.ckpt
```

### Run inference

Our inference script allows for DDP. By default we sample 10 sequences per
length between 60 and 128. Samples are stored as PDB files as well as the
trajectories. We do not include evaluation code using ProteinMPNN and ESMFold
but this should be easy to set-up if one looks at the [FrameDiff codebase](https://github.com/jasonkyuyim/se3_diffusion).

```bash
python -W ignore experiments/inference_se3_flows.py
```

# Responsible AI FAQ
- What is FrameFlow?
  - FrameFlow is a deep neural network that models 3D protein structures.
- What can FrameFlow do?
  - By sampling from FrameFlow, you can obtain a description of the positions and orientations of the backbone atoms in a protein.
- What is/are FrameFlow’s intended use(s)?
  - FrameFlow is intended for research purposes only, for the machine learning for structural biology community.
- How was FrameFlow evaluated? What metrics are used to measure performance?
  - FrameFlow was evaluated on how novel, designable and diverse the protein structures sampled from FrameFlow were. 
- What are the limitations of FrameFlow? How can users minimize the impact of FrameFlow’s limitations when using the system?
  - FrameFlow has not been tested by real-world experiments to see if the proteins it samples are actually designable. FrameFlow should be used for research purposes only.
- What operational factors and settings allow for effective and responsible use of FrameFlow?
  - FrameFlow should be used for research purposes only.

# Contributing
This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the Microsoft Open Source Code of Conduct. For more information see the Code of Conduct FAQ or contact opencode@microsoft.com with any additional questions or comments.

# Trademarks
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
