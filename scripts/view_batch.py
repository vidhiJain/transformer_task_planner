from pathlib import Path
import hydra
from temporal_task_planner.view_generated_sessions import init_sim, session_viewer
from glob import glob


@hydra.main(config_path="../config", config_name="view")
def main(config):
    sim = init_sim()
    session_jsons = glob(Path(config["dirpath"], config["regex_session"]).as_posix())
    for session in session_jsons:
        config["session"] = session
        session_viewer(sim, config)


if __name__ == "__main__":
    main()
