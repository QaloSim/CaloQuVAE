# CaloQuVAE
This repository is a lighter and faster version of the CaloQVAE repo. It trims 10 minutes off each epoch, ultimately trimming ~4hrs per model.

## ToDo
- [x] For datasets, the plan is to have 1 config file per dataset.
- [x] For architectures, the plan is to make it more horizontal, allowing us to combine different architectures easier. One .py file per model (encoder/decoder).
- [x] Also we want to add the feature where when one loads a model, it loads the corresponding config of the model. So no need to match the config with the loaded model.
- [x] Also fix the RBM issue when increasing size.
- [x] Also figure out what we wanna plot, and make it easier to turn off/on certain plots
- [x] Add AE stand-alone training 
- [ ] Add Plots for CaloChallenge
- [ ] Add RBM stand-alone training
- [ ] Add D-Wave sampling and inverse Temp estimation
- [ ] Add KPD and FPD estimation
- [ ] Add Transformer architecture

## A Quantum-assisted Deep Generative Particle-Calorimeter Surrogate
![](https://github.com/QaloSim/CaloQuVAE/blob/main/infographic.png)

## Overview
### Repository Structure


| Directory        | Content    | 
| ------------- |:-------------| 
| `configs/`      | Configuration files | 
| `data/` | Data manager and loader |
| `engine/`  | Training loops. |
| `models/` | Core module, includes definitions of all models.  |
| `notebooks/` | Standalone experimentation notebooks. |
| `scripts/` | Steering scripts includes one to run - run.py|
| `utils/` | Helper functionalities for core modules (plotting etc.) |

### Input Data

|  Dataset | Location |
| ------------- | ------------- |
| CaloChallenge  | [DOI](https://zenodo.org/records/6366271) |


## Setup
```
git clone git@github.com:QaloSim/CaloQuVAE.git
cd CaloQuVAE
```

### Installation
#### Via Virtual Environment and pip
Initial package setup:
```
source source.me
python -m pip install -r requirements.txt
```

### After Installation
After the initial setup, simply navigate to the package directory and run

```
source source.me
```
Sources the virtual environment and appends to `PYTHONPATH`.

## How To...

### ...configure models
We're currently using Hydra for config management. The top-level file is `config.yaml`. For more info on Hydra, click [here](https://hydra.cc/docs/tutorials/intro/)

### ...run models
```
python scripts/run.py
```

## Technical note
When running on the TRIUMF ml machine, `DISPLAY` variable must be unset (it can be set by forwarding X11 when creating the ssh session), as it creates an unwanted dependency with a QT library. 

### References
[1] Toledo-Marin JQ, Gonzalez S, Jia H, Lu I, Sogutlu D, Abhishek A, Gay C, Paquet E, Melko R, Fox GC, Swiatlowski M., Fedorko W Conditioned quantum-assisted deep generative surrogate for particle-calorimeter interactions. npj Quantum Inf 11, 114 (2025). https://doi.org/10.1038/s41534-025-01040-x

[2] Fedorko WT, Toledo-Marín JQ, Fox GC, Gay CW, Jia H, Lu I, Melko R, Paquet E, Sogutlu D, Swiatlowski MJ. Quantum-Assisted Generative AI for Simulation of the Calorimeter Response.

### Citation
If you use this package in a publication, please cite:
* Toledo-Marin JQ, Gonzalez S, Jia H, Lu I, Sogutlu D, Abhishek A, Gay C, Paquet E, Melko R, Fox GC, Swiatlowski M., Fedorko W Conditioned quantum-assisted deep generative surrogate for particle-calorimeter interactions. npj Quantum Inf 11, 114 (2025). https://doi.org/10.1038/s41534-025-01040-x