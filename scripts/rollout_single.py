import os
from pathlib import Path
import hydra
from temporal_task_planner.rollout import session_rollout

"""Similar to rollout batch 
Ignores the session start id and end id tokens 
 and doesn't need to call batch_generate

Saves: generated session json for expert policy rollout at
    Path(config["dirpath"], "tmp.json").as_posix()
"""


@hydra.main(config_path="../config", config_name="rollout")
def main(config):
    curdir = hydra.utils.get_original_cwd()
    for subdir in config["dirpath"].split("/"):
        curdir = Path(curdir, subdir)
        if not os.path.exists(curdir):
            os.mkdir(curdir)
    env = hydra.utils.instantiate(config["envs"])
    obs = env.reset()
    policy = hydra.utils.instantiate(config["policy"])
    policy.reset(env)
    savepath = Path(config["dirpath"], "tmp.json").as_posix()
    session_rollout(obs, env, policy, savepath)


if __name__ == "__main__":
    main()
