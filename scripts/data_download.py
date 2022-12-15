"""
Downloads the dataset from wandb 
Stores in artifacts/<dataset_name>:<version>/...
"""
import wandb

run = wandb.init()
datasets = [
    "dishwasher_arrange/dataset/full-visible-single-pref:latest",
    "dishwasher_arrange/dataset/partial-visible-single-pref:latest",
]
for dataset in datasets:
    artifact = run.use_artifact(dataset, type="raw-data")
    artifact_dir = artifact.download()
