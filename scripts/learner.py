import os
import hydra
import wandb
import json
from hydra.utils import instantiate, get_original_cwd
from pathlib import Path


def construct_run_name(context_history, batch_size):
    """Custom name format for comparison plots"""
    return f"cw{context_history}_b{batch_size}"  # f"cw-{cfg['config']['context_history']}_b-{cfg['config']['batch_size']}"


@hydra.main(config_path="../config", config_name="learner")
def main(cfg):
    cfg.pkg_root = hydra.utils.get_original_cwd()
    learner = hydra.utils.instantiate(cfg.learner)
    # learner.init_wandb(cfg)
    id = cfg.id if cfg.id != "" else wandb.util.generate_id()
    cfg.id = id
    context_history = cfg.context_history
    if cfg.batch_size == 0:
        batch_size = int(cfg.num_targets_per_step) // (int(cfg.context_history) + 1)
    else:
        batch_size = cfg.batch_size
    name = construct_run_name(context_history, batch_size)
    with wandb.init(
        id=id,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=cfg,
        save_code=False,
        name=name,
        resume="allow",
    ) as run:
        # print(f"cfg.run.dir = {cfg.run.dir}")
        # write to a file
        runid_dict = {
            id: f"{cfg.data_name}-v{cfg.data_version}_cw{cfg.context_history}_b{cfg.batch_size}"
        }
        with open(Path(hydra.utils.get_original_cwd(), "runid.json"), "a") as f:
            json.dump(runid_dict, f)
        learner.train()


if __name__ == "__main__":
    main()
