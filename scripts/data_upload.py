import wandb

with wandb.init(project="dataset", entity="dishwasher_arrange") as run:
    data = wandb.Artifact("full-visible-single-pref", type="raw-data")
    data.add_dir("data/pickplace/full_visible")
    run.log_artifact(data)
