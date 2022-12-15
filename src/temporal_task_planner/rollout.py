from pathlib import Path
from typing import Dict, List, Tuple
from temporal_task_planner.data_structures.state import State
from temporal_task_planner.data_structures.action import Action
from temporal_task_planner.envs.dishwasher_env import DishwasherLoadingTaskEnv
from temporal_task_planner.policy.abstract_policy import Policy


def session_rollout(
    obs: State, env: DishwasherLoadingTaskEnv, policy: Policy, savepath: str = None
) -> Dict:
    """Rollout the policy on the env, and save the session json
    Returns:
        List of actions taken
    """
    actions = []
    done = False
    while not done:
        act = policy.get_action(obs)
        # print(f"\nAction {env.step_counter}: {act}")
        obs, rew, done, info = env.step(act)
        actions.append(act)
    # print("Actions: ", len(actions))
    session_dict = env.env_utils.get_session_dict()
    if savepath is not None:
        env.env_utils.save_session(session_dict, savepath)
        print(f"Saved policy at : {savepath}")
    info = {
        "placed_utensils": session_dict["session"]["userActions"][-1]["placedUtensils"],
        "actions": actions,
    }
    return info


def batch_generate(
    session_id_start: int,
    session_id_end: int,
    env: DishwasherLoadingTaskEnv,
    policy: Policy,
    dirpath=str,
):
    """env and policy completely reset in session rollout!"""
    for i in range(session_id_start, session_id_end):
        savepath = Path(dirpath, f"sess_{i}.json").as_posix()
        obs = env.reset()
        policy.reset(env=env, obs=obs)
        session_rollout(obs, env, policy, savepath)
