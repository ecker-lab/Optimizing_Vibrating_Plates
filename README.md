#  Guided Flow Matching Design Optimization for Minimizing Structural Vibrations
![method_figure](https://github.com/user-attachments/assets/ff0f586e-b3f3-4ede-a040-4b7eb2ad218e)

Structural vibrations are a source of unwanted noise in engineering systems like cars, trains or airplanes. Minimizing these vibrations is crucial for improving passenger comfort. This work presents a novel design optimization approach based on guided flow matching for reducing vibrations by placing beadings (indentations) in plate-like structures. Our method integrates a generative flow matching model and a surrogate model trained to predict structural vibrations. During the generation process, the flow matching model pushes towards manufacturability while the surrogate model pushes to low-vibration solutions. The flow matching model and its training data implicitly define the design space, enabling a broader exploration of potential solutions as no optimization of manually-defined design parameters is required. We apply our method to a range of differentiable optimization objectives, including direct optimization of specific eigenfrequencies through careful construction of the objective function. Results demonstrate that our method generates diverse and manufacturable plate designs with reduced structural vibrations compared to designs from random search, a criterion-based design heuristic and genetic optimization.

Preprint available from [arxiv](https://arxiv.org/abs/2506.15263).
## Repository structure

This repository mainly contains the following elements:

- (Unguided) flow matching code to generate novel beading patterns in [`plate_optim/flow_matching`](./plate_optim/flow_matching).
- Surrogate / regression code to predict dynamic responses of plates with beading patterns in [`plate_optim/regression`](./plate_optim/regression).
- A combination of the regression resulting in a guided flow matching solver in [`plate_optim/guided_flow.py`](./plate_optim/guided_flow.py) and [`plate_optim/utils/guidance.py`](./plate_optim/utils/guidance.py).
- Loss functions to guide towards low vibrations in [`plate_optim/utils/guidance.py`](./plate_optim/utils/guidance.py).
- Metrics to assess the manufacturability of generated plate designs in [`plate_optim/metrics`](./plate_optim/metrics).
- A procedural beading pattern design generation method, with no direct relation to vibrations,  in [`plate_optim/pattern_generation.py`](./plate_optim/pattern_generation.py).
- A workflow to evaluate beading patterns via FE computation [`notebooks/simulate_plate_via_fem.ipynb`](notebooks/simulate_plate_via_fem.ipynb)

If you would like to try out our method, we recommend getting started with the notebook saved in [`notebooks/run_guided_fm_design_optimization.ipynb`](notebooks/run_guided_fm_design_optimization.ipynb)
<a target="_blank" href="https://colab.research.google.com/github/ecker-lab/Optimizing_Vibrating_Plates/blob/main/notebooks/run_guided_fm_design_optimization.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>  
</a> 

## Data and model checkpoints

Our dataset consisting of beading patterns and associated numerical simulations is available from [here](https://doi.org/10.25625/XMYQHO).
The individual files can be downloaded via e.g. ``wget``:

- D500_edit.h5 https://data.goettingen-research-online.de/api/access/datafile/125125
- D50k_edited.h5 https://data.goettingen-research-online.de/api/access/datafile/125180
- moments_D50k.pt https://data.goettingen-research-online.de/api/access/datafile/125124
- flow_matching_0331.ckpt https://data.goettingen-research-online.de/api/access/datafile/125254
- regression_075_noise_0312.ckpt https://data.goettingen-research-online.de/api/access/datafile/125255
- regression_no_noise_0526.ckpt https://data.goettingen-research-online.de/api/access/datafile/125253

### Data generation and evaluation 

Our numerical simulations are performed with [elpaso](https://akustik.gitlab-pages.rz.tu-bs.de/elPaSo-Core/main/intro.html), which is a fast and flexible simulation tool developed specifically for vibroacoustics. Elpaso is not provided as part of this package. The numerical simulations can also be performed with other finite-element solvers. As an example, we provide directions how to evaluate the plate model with Abaqus [`notebooks/simulate_plate_via_fem.ipynb`](notebooks/simulate_plate_via_fem.ipynb).


## Getting started

The video is an example of the optimization process for a beading pattern. It shows the beading pattern starting from random noise, the gradient signal from the regression model and the prediction from the flow matching model (from left to right). The red cross marks the loading position.

https://github.com/user-attachments/assets/2f9ef409-817f-49ce-86cd-58cc5ccdf3ed


If you would like to try out our design optimization method, we recommend getting started with with the notebook saved in [`notebooks/run_guided_fm_design_optimization.ipynb`](notebooks/run_guided_fm_design_optimization.ipynb)
<a target="_blank" href="https://colab.research.google.com/github/ecker-lab/Optimizing_Vibrating_Plates/blob/main/notebooks/run_guided_fm_design_optimization.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>  
</a> 


If you would like to train models yourself, first git clone this repository and then set up the environment following the ``environment.yml`` file:

```
conda create -f environment.yml
pip install -e .
```

To specify where you want to save the data, models, etc., you can change the paths in [plate_optim/project_directories.py](plate_optim/project_directories.py)
This repository employs wandb for logging. To be able to use it you must have an account and login: ``wandb login``

**Evaluation:** This repository does not contain code for performing numerical simulations. We perform simulations with [elpaso](https://akustik.gitlab-pages.rz.tu-bs.de/elPaSo-Core/main/intro.html). We provide a notebook to evaluate plates with beading patterns using the FEM software Abaqus [`notebooks/simulate_plate_via_fem.ipynb`](notebooks/simulate_plate_via_fem.ipynb).


## Training and optimization

All scripts require that the datasets are downloaded and the paths are correctly set up as described in ``Getting started``.

### Training surrogate / regression models
Run one of the scripts in [scripts/regression](scripts/regression), e.g.:
``source scripts/regression/50k15_075noise.sh``

### Training flow matching models
Run the script in [scripts/flow_matching](scripts/flow_matching):
``source scripts/flow_matching/standard.sh``


### Optimization 

Performing design optimization requires already trained models, which can be downloaded as described in the data section of this readme. 
To run flow matching for the 100-200 Hz frequency range and simply supported plates, run the following:

```
source scripts/range100_200.sh
source scripts/setting_a_args.sh
source scripts/experiment_scripts/run_flow_matching.sh
```

The first two scripts define environmental variables for the frequency range, the boundary condition, the model checkpoints, etc.. 
The third script actually calls [`plate_optim/guided_flow.py`](./plate_optim/guided_flow.py) with the set variables. 
Before running these scripts, you probably need to adjust the paths inside ``setting_a_args.sh`` to actually point to the model checkpoints 
and you might want to change the directory where results are saved in ``run_flow_matching.sh``. Alternatively, the relevant environmental variables can be provided directly in the terminal like this:


```
min_freq=100
max_freq=200
n_freqs=101
freqs='100_200'
extra_conditions='[0,0.34444444444444444,0.35]' # 0, 0.31 / 0.9, 0.21 / 0.6 # clamping, loading position x, loading position y
flow_matching_path='path'
regression_path_noise='path'
setting='a'
source scripts/experiment_scripts/run_flow_matching.sh
```

This block gives an overview over the available scripts:
<pre>
scripts|
|____range100_200.sh # specify frequency range 100-200 Hz
|____range200_250.sh 
|____setting_a_args.sh # specify boundary condition setting according to paper
|____setting_b_args.sh
|____experiment_scripts # after specifying boundary condition and frequency range run one of the optimization methods
| |____run_random_search.sh # baseline method
| |____run_flow_matching.sh # standard guided flow matching
| |____run_fm_first_peak.sh # flow matching for optimizing the first eigenfrequency   
| |____run_genetic_opt.sh # baseline method
| |____further_scripts.sh # other scripts for ablation studies
</pre>


## License 

Most of our code is MIT licensed. The unguided flow matching code in this repository builds mainly upon the [flow matching package](https://github.com/facebookresearch/flow_matching) which is CC BY-NC licensed and therefore has the same license. This license applies to the code within the [plate_optim/flow_matching](plate_optim/flow_matching) directory. The rest of our repository is released under the MIT license. Please note, that the published data is not licensed under MIT, as detailed [here](https://doi.org/10.25625/XMYQHO).


## Citation

```
@article{delden2025minimizing,
  author={van Delden, Jan and Schultz, Julius and Rothe, Sebastian and Libner, Christian and Langer, Sabine C. and L{\"u}ddecke, Timo},
  title={Minimizing Structural Vibrations via Guided Flow Matching Design Optimization},
  journal={arXiv preprint arXiv:2506.15263},
  year={2025},
}
```

The regression model employed in this work builts upon [our previous project](https://github.com/ecker-lab/Learning_Vibrating_Plates) on predicting structural vibrations.
```
@inproceedings{delden2024vibrations,
  title={Learning to Predict Structural Vibrations},
  author={van Delden, Jan and Schultz, Julius and Blech, Christopher and Langer, Sabine C and L{\"u}ddecke, Timo},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=i4jZ6fCDdy}
}
```

## Acknowledgements

Our work is in part built upon the following wonderful repositories:
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq)
- [flow_matching package](https://github.com/facebookresearch/flow_matching)
