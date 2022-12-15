import hydra
import os
from pathlib import Path

from temporal_task_planner.rollout import batch_generate


@hydra.main(config_path="../config", config_name="rollout")
def main(config):
    curdir = hydra.utils.get_original_cwd()
    for subdir in config["dirpath"].split("/"):
        curdir = Path(curdir, subdir)
        if not os.path.exists(curdir):
            os.mkdir(curdir)

    batch_generate(
        config["session_id_start"],
        config["session_id_end"],
        hydra.utils.instantiate(config["envs"]),
        hydra.utils.instantiate(config["policy"]),
        config["dirpath"],
    )


if __name__ == "__main__":
    main()
