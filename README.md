# FrameFlow

FrameFlow is a SE(3) flow matching method for protein backbone generation and motif-scaffolding.
The method is described in two papers:

* [Improved motif-scaffolding with SE(3) flow matching](https://arxiv.org/abs/2401.04082)
* [Fast protein backbone generation with SE (3) flow matching](https://arxiv.org/abs/2310.05297)

Unconditional protein backbone generation:

![frameflow-uncond](https://github.com/microsoft/flow-matching/blob/main/media/unconditional.gif)

Motif condifioned scaffold backbone generation:

![frameflow-cond](https://github.com/microsoft/flow-matching/blob/main/media/scaffolding.gif)

If you use this work (or code), then please cite the first paper (citing both would make me happier :).

```bash
@article{yim2024improved,
  title={Improved motif-scaffolding with SE(3) flow matching},
  author={Jason Yim and Andrew Campbell and Emile Mathieu and Andrew Y. K. Foong and Michael Gastegger and José Jiménez-Luna and Sarah Lewis and Victor Garcia Satorras and Bastiaan S. Veeling and Frank Noé and Regina Barzilay and Tommi S. Jaakkola},
  journal={Transactions on Machine Learning Research},
  year={2024}
}

@article{yim2023fast,
  title={Fast protein backbone generation with SE (3) flow matching},
  author={Yim, Jason and Campbell, Andrew and Foong, Andrew YK and Gastegger, Michael and Jim{\'e}nez-Luna, Jos{\'e} and Lewis, Sarah and Satorras, Victor Garcia and Veeling, Bastiaan S and Barzilay, Regina and Jaakkola, Tommi and others},
  journal={arXiv preprint arXiv:2310.05297},
  year={2023}
}
```

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

```bash
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

```bash
weights
├── config.yaml
└── published.ckpt
```

### Unconditional sampling

Our inference script allows for DDP. By default we sample 10 sequences per
length [70, 100, 200, 300]. Samples are stored as PDB files as well as the
trajectories. We do not include evaluation code using ProteinMPNN and ESMFold
but this should be easy to set-up if one looks at the [FrameDiff codebase](https://github.com/jasonkyuyim/se3_diffusion).

```bash
# Single GPU
python -W ignore experiments/inference_se3_flows.py -cn inference_unconditional

# Multi GPU
python -W ignore experiments/inference_se3_flows.py -cn inference_unconditional inference.num_gpus=2
```

### Motif-scaffolding

Like unconditional sampling, multi-GPU DDP is supported.
We support the RFdiffusion motif-scaffolding benchmark as described in [Supp. Methods Table 9 of Watson et al](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-023-06415-8/MediaObjects/41586_2023_6415_MOESM1_ESM.pdf).
We do not include motif 6VW1 since it involves multiple chains that FrameFlow cannot handle.
Benchmarking works by reading the contig settings in `motif_scaffolding/benchmark.csv`.
Consider target `3IXT` in the CSV.

| target | contig                        | length | motif_path                              |
|--------|-------------------------------|--------|-----------------------------------------|
| 3IXT   | 10-40,P254-277,10-40          | 50-75  | ./motif_scaffolding/targets/3IXT.pdb    |

Explanation of each column:

* `target`: unique identifier for the motif.
* `contig`: same contig syntax as RFdiffusion to specify motifs from a PDB file and scaffolds to sample.
See the [motif-scaffolding section in the RFdiffusion README](https://github.com/RosettaCommons/RFdiffusion?tab=readme-ov-file#motif-scaffolding) for more information.
* `length`: randomly sampled total scaffold length.
* `motif_path`: path to PDB file with motif.

> [!NOTE]  
> To specify your own motif, follow the syntax in `benchmark.csv` and point `samples.csv_path` to the your custom CSV with motif-scaffolding tasks.

To run motif-scaffolding, we specify the settings in `configs/inference_scaffolding.yaml`.
See `inference.samples` for different sampling settings.

```bash
# Single GPU
python -W ignore experiments/inference_se3_flows.py -cn inference_scaffolding

# Multi GPU
python -W ignore experiments/inference_se3_flows.py -cn inference_scaffolding inference.num_gpus=2
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
