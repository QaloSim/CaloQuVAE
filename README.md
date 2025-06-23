# CaloQVAE

Calo4pQVAE: A Quantum-assisted Deep Generative Particle-Calorimeter Surrogate
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
[1] Toledo-Marin JQ, Gonzalez S, Jia H, Lu I, Sogutlu D, Abhishek A, Gay C, Paquet E, Melko R, Fox GC, Swiatlowski M., Fedorko W Conditioned quantum-assisted deep generative surrogate for particle-calorimeter interactions. arXiv preprint arXiv:2410.22870. 2024 Oct 30.

[2] Fedorko WT, Toledo-Marín JQ, Fox GC, Gay CW, Jia H, Lu I, Melko R, Paquet E, Sogutlu D, Swiatlowski MJ. Quantum-Assisted Generative AI for Simulation of the Calorimeter Response.
