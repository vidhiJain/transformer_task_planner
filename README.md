# Transformer Task Planner
Vidhi Jain, Yixin Lin, Eric Undersander, Yonatan Bisk and Akshara Rai, Transformers are Adaptable Task Planners, 6th Conference on Robot Learning (CoRL 2022).

[OpenReview](https://openreview.net/forum?id=Eal_lL08v_l) | 
[Website](https://sites.google.com/andrew.cmu.edu/ttp/home)


[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Setup

### 1. Create conda environment
```bash
conda env create -f environment.yml
```
### 2. Activate conda environment 
- using https://direnv.net/ ...
```bash 
direnv allow
```
- or manually  activate
```bash 
conda activate temporal_task_planner
```
### 3. Install [PyTorch](https://pytorch.org/) according to your system requirements.
For example: MacOS installation, cpu only
```bash 
# MacOS Conda binaries are for x86_64 only, for M1 please use wheels
conda install pytorch -c pytorch
```

### 4. Init submodules 
```bash
git submodule update --init --recursive
```

### 5. Build custom habitat-sim
If you are installing habitat-sim for the first time, you might need some additional libraries.
Follow instructions from [habitat_sim: BUILD_FROM_SOURCE](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md).
```bash 
cd third_party/habitat-sim
./build.sh --bullet --with-cuda --headless  # this might take a while...
cd -
```

### 6. Install temporal_task_planner
```bash 
pip install -e . 
```

### 7. [Optional for sweeps on cluster] Install hydra/launcher submitit slurm 
```bash
pip install hydra-submitit-launcher --upgrade
```

### 8. Add habitat-sim to your python path
```bash
export PYTHONPATH=:$PWD/third_party/habitat-sim
echo $PYTHONPATH
```

### 9. [Optional if using cuda] Set the CUBLAS CONFIG for fixed seed
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```
--

## Data 
### 1. Kitchen assets

Download from scratch and unzip it:
- [data_v3.zip](https://drive.google.com/drive/folders/1A24fBE077AFecZVRe9YPgTdu6fDA0EaC)
- [scenes.zip](https://drive.google.com/file/d/1VFQqvIqqqBQFfcype0guzAdpcez2brOo/view?usp=sharing)

### 2. Pre-generated Demonstrations 
Session jsons can be downloaded from wandb. Request access by emailing `vidhij@andrew.cmu.edu`.
```bash 
python scripts/data_download.py
```
This creates `artifacts/` folder containing 
`full-view-single-pref:latest` and `partial-view-single-pref:latest` datasets, where each contains train, val and test jsons.

---

## Run 
All the files in `scripts/` folder can be run as 
```
python scripts/<filename>.py
```
 Scripts 1-3 are dependent on hydra yaml config files.

#### 0. To download data: 
```python scripts/data_download.py```
This downloads the latest single preference data for full and partial visibility scenarios.

#### 1. To rollout expert policy: 
```python scripts/rollout_batch.py ```
You need to provide config parameters like `dirpath, session_id_start, session_id_end`.

#### 2. To generate session videos: 
```python scripts/view_batch.py```
You need to provide config parameters like `dirpath, session_id_start, session_id_end`.

#### 3. To train the model: 
```python scripts/learner.py```
You need to provide config parameters like `pick_only, context_history, data_name, data_version`

