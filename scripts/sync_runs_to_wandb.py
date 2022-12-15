import os
from glob import glob
from pathlib import Path
import subprocess

def main(args):
    rootpath = Path(
        args.project_name, 
        f"{args.data_name}-v{args.data_version}", 
    )
    if not os.path.exists(rootpath):
        print(f"{rootpath} does not exists!!!")
    pathlists = glob(f"{rootpath}/*/*/wandb/run-*")
    # print(pathlists)
    for path in pathlists:
        wandb_id = path.split("-")[-1] 
        subprocess.run(f"wandb sync --id {wandb_id} {path}".split(" "))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--project_name")
    parser.add_argument('-n', "--data_name")
    parser.add_argument('-v', "--data_version")
    args, _ = parser.parse_known_args()
    main(args)