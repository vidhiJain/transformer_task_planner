import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import gym
import json
import numpy as np
from requests import session
import numpy.typing as npt
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score
import textdistance
import wandb
from transformers import PreTrainedModel

from temporal_task_planner.constants.gen_sess_config.lookup import (
    category_vocab,
    special_instance_vocab,
    category_dict,
)
from temporal_task_planner.policy.expert_policy import ExpertPickOnlyPreferencePolicy
from temporal_task_planner.policy.learned_policy import LearnedPolicy
from temporal_task_planner.rollout import session_rollout
from temporal_task_planner.utils.data_structure_utils import get_actions_from_session
from temporal_task_planner.utils.extract_from_session import (
    get_placed_utensils_in_useraction,
)
from temporal_task_planner.utils.gen_sess_utils import load_session_json

# Fix this!
from temporal_task_planner.trainer.dataset import DishwasherArrangeSavedSession, pad_fn


class Logger:
    def __init__(
        self,
        env: Any,  # gym.Env,
        session_paths: List[str],
        config: Dict,
        model: PreTrainedModel,
        criterion: torch.nn.Module,
        name: str,
        pick_only: bool = True,
        max_evals: int = 10,
        savefolder: str = "rollouts",
    ):

        self.env = env
        self.session_paths = session_paths  # deepcopy(self.dataset.session_paths)

        self.savefolder = savefolder

        if not os.path.exists(self.savefolder):
            os.mkdir(self.savefolder)

        self.expert_rollout_info = []
        for session_id in range(max_evals):
            # read the session file
            data = load_session_json(self.session_paths[session_id])
            self.expert_rollout_info.append(
                {
                    "placed_utensils": get_placed_utensils_in_useraction(
                        data, useraction_idx=-1
                    ),
                    "actions": get_actions_from_session(data),
                }
            )
        #     self.env.reset(ref_sess=self.session_paths[session_id])
        #     self.expert_rollout_info.append(
        #         session_rollout(
        #             self.env,
        #             expert_policy,
        #             savepath=None
        #             # Path(self.savefolder, "expert_rollout.json").as_posix(),
        #         )
        #     )

        # init env with a session from the session paths
        # TODO: compute over all session paths

        self.config = config
        self.device = self.config["device"]
        self.context_history = self.config["context_history"]

        self.model = model
        self.criterion = criterion
        self.name = name
        self.pick_only = pick_only
        if not self.pick_only:
            self.pickplacelogs = dict(
                PickAcc=[],
                PlaceAcc=[],
            )
        self.max_evals = max_evals

        self.categ_attr = ["timestep", "category_token"]
        self.cont_attr = ["category-3dBB", "pose"]
        self.input_attr = self.categ_attr + self.cont_attr
        self.rollout_metrics = ["TPE", "BPE", "PE", "SPL", "LC"]
        self.main_log = None  # self.init_main_log()
        self.sess_log = None

    def init_main_log(self, to_rollout: bool = False):
        """
        Records Loss, Acc, etc per session
        """
        main_log = dict(
            Loss=[],
            Acc=dict(symbol=[], attribute={key: [] for key in self.categ_attr}),
            # Precision=dict(attribute={key: [] for key in self.categ_attr}),
            # Recall=dict(attribute={key: [] for key in self.categ_attr}),
            # AvgL2=dict(attribute={key: [] for key in self.cont_attr}),
        )
        if to_rollout:
            main_log.update(
                dict(
                    TPE=[],
                    BPE=[],
                    PE=[],
                    SPL=[],
                    LC=[],
                )
            )
        return main_log

    def __call__(
        self, model, step: int, to_rollout: bool, *args: Any, **kwds: Any
    ) -> Any:
        """Calculates metrics per session and averages them"""
        print(f"Metric computation: to_rollout={to_rollout}")
        self.main_log = self.init_main_log(to_rollout)
        # New params of the Model, in eval mode
        self.model = model
        self.model.eval()
        if not os.path.exists("rollouts"):
            os.mkdir("rollouts")
        # Sample `max_evals` sessions from session_paths
        for session_id in range(self.max_evals):
            self.get_per_step_metrics(session_id)
            if to_rollout:
                savepath = (
                    f"rollouts/{self.name}_step-{step}_session_id-{session_id}.json"
                )
                self.get_rollout_metrics(session_id, savepath)
        # Format metrics and upload to wandb
        metrics = self.get_avg_log(to_rollout)
        self.record(metrics, step=step)
        return metrics

    def init_sess_log(self) -> Dict:
        """Aggregates the target and predicted lists for
        sklearn metrics
        """
        sess_logs =  dict(
            losses=[],
            y_pred=[],
            y_true=[],
            x_pred={key: [] for key in self.input_attr},
            x_true={key: [] for key in self.input_attr},
        )
        return sess_logs

    def get_attr_based_predictions(self, inputs: Dict, targets: torch.Tensor) -> Dict:
        """Processes the attribute dictionary for input instances"""
        targets = targets.reshape(len(inputs["timestep"]), -1)
        _x = {key: [] for key in self.input_attr}
        for attr_type in ["timestep", "pose", "category_token"]:
            for i in range(len(targets)):
                _x[attr_type] += deepcopy(
                    inputs[attr_type][i, targets[i, :]].cpu().detach().numpy().tolist()
                )
        # category : 3dBB
        for i in range(len(targets)):
            _x["category-3dBB"] += deepcopy(
                inputs["category"][i, targets[i, :]].cpu().detach().numpy().tolist()
            )
        return _x

    def accumulate(
        self,
        loss: int,
        y_true: npt.ArrayLike,
        y_pred: npt.ArrayLike,
        x_true: Dict[str, List],
        x_pred: Dict[str, List],
    ) -> None:
        """
        Args:
            loss, inputs, out, targets : torch Tensor
        Return:
            self.sess_log : dict of (dict of) list (No numpy, no torch)
        """
        self.sess_log["losses"].append(loss.item())
        self.sess_log["y_true"] += y_true.tolist()
        self.sess_log["y_pred"] += y_pred.tolist()
        for key in self.sess_log["x_true"].keys():
            self.sess_log["x_true"][key] += x_true[key]
            self.sess_log["x_pred"][key] += x_pred[key]
        return self.sess_log

    def update_main_log(self) -> None:
        """Update loss, acc and other metrics of per step predictions"""
        self.main_log["Loss"].append(np.mean(self.sess_log["losses"]))
        self.main_log["Acc"]["symbol"].append(
            accuracy_score(self.sess_log["y_true"], self.sess_log["y_pred"])
        )
        for key in ["category_token"]:  # self.categ_attr:
            self.main_log["Acc"]["attribute"][key].append(
                accuracy_score(
                    self.sess_log["x_true"][key], self.sess_log["x_pred"][key]
                )
            )
            # self.main_log["Precision"]["attribute"][key].append(
            #     precision_score(
            #         self.sess_log["x_true"][key],
            #         self.sess_log["x_pred"][key],
            #         average="weighted",
            #         zero_division=0,
            #     )
            # )
            # self.main_log["Recall"]["attribute"][key].append(
            #     recall_score(
            #         self.sess_log["x_true"][key],
            #         self.sess_log["x_pred"][key],
            #         average="weighted",
            #         zero_division=0,
            #     )
            # )
        # for key in self.cont_attr:
        #     avgL2 = np.mean(
        #         np.linalg.norm(
        #             np.array(self.sess_log["x_true"][key])
        #             - np.array(self.sess_log["x_pred"][key]),
        #             axis=-1,
        #             ord=2,
        #         )
        #     )
        #     self.main_log["AvgL2"]["attribute"][key].append(avgL2)
        return

    def get_avg_log(self, to_rollout: bool = False) -> Dict:
        """
        Recording mean of logs per session
        TODO: add std value too for error bar.
        """
        metrics = {}
        metrics.update(
            {
                f"{self.name}/Loss": np.mean(self.main_log["Loss"]),
                f"{self.name}/Acc/symbol": np.mean(self.main_log["Acc"]["symbol"]),
            }
        )
        for attr in ["category_token"]:
            metrics.update(
                {
                    f"{self.name}/Acc/{attr}": np.mean(
                        self.main_log["Acc"]["attribute"][attr]
                    ),
                    #     f"{self.name}/Precision/{attr}": np.mean(
                    #         self.main_log["Precision"]["attribute"][attr]
                    #     ),
                    #     f"{self.name}/Recall/{attr}": np.mean(
                    #         self.main_log["Recall"]["attribute"][attr]
                    #     ),
                }
            )
        # for attr in self.cont_attr:
        #     metrics.update(
        #         {
        #             f"{self.name}/AvgL2/{attr}": np.mean(
        #                 self.main_log["AvgL2"]["attribute"][attr]
        #             )
        #         }
        #     )
        if to_rollout:
            for rollout_metric in self.rollout_metrics:
                metrics.update(
                    {
                        f"{self.name}/{rollout_metric}": np.mean(
                            self.main_log[f"{rollout_metric}"]
                        )
                    }
                )
        return metrics

    def log_true_pred_tokens(self, step: int) -> List:
        """Processes the values of True and Predicted vocab tokens
        per step (along with loss at that step) for failure case analysis"""
        target_token_names = self.sess_log["x_true"]["category_token"]
        pred_token_names = self.sess_log["x_pred"]["category_token"]

        data = {
            "Step": step,
            "Loss": self.main_log["Loss"],
            "True": [
                category_vocab.index2word(target_token_names[i])
                for i in range(len(target_token_names))
            ],
            "Predicted": [
                category_vocab.index2word(pred_token_names[i])
                for i in range(len(pred_token_names))
            ],
        }
        return list(data.values())

    @torch.no_grad()
    def get_per_step_metrics(self, session_id: int) -> None:
        """Inits the dataset class wrapper to process each session
        independently for the computing metrics"""
        self.sess_log = self.init_sess_log()
        assert session_id < len(self.session_paths), print(
            session_id, len(self.session_paths)
        )
        session_dataset = DishwasherArrangeSavedSession(
            session_paths=[self.session_paths[session_id]],
            context_history=self.context_history,
            num_sessions_limit=1,
            pick_only=self.pick_only,
        )
        dataloader = DataLoader(
            session_dataset, batch_size=len(session_dataset) + 1, collate_fn=pad_fn
        )
        for inputs, targets in dataloader:
            out = self.model(**inputs, device=self.device)
            loss = self.criterion(out, targets)
            y_true = (
                torch.cat(targets["action_instance"], dim=0).cpu().numpy()
            )  # .tolist()
            y_pred = (
                torch.argmax(out["pick"], dim=-1).reshape(-1).cpu().detach().numpy()
            )  # .tolist()
            x_true = self.get_attr_based_predictions(inputs, y_true)
            x_pred = self.get_attr_based_predictions(inputs, y_pred)
            self.accumulate(loss, y_true, y_pred, x_true, x_pred)
            if not self.pick_only: 
                # pick-place acc metrics
                y_true = (
                    torch.stack(targets["action_instance"], dim=0).cpu().numpy()
                )  # .tolist()
                y_pred = (
                    torch.argmax(out["pick"], dim=-1).reshape(-1, 2).cpu().detach().numpy()
                )  # .tolist()
                self.pickplacelogs["PickAcc"] = accuracy_score(
                    y_true[:, 0], y_pred[:, 0]
                )
                self.pickplacelogs["PlaceAcc"] = accuracy_score(
                    y_true[:, 1], y_pred[:, 1]
                )
                print(self.pickplacelogs)
        self.update_main_log()

    def per_step_predictions(self) -> Tuple[np.array, np.array]:
        # log contains inp-out-tgt & losses from every timestep in given sessions.
        y_pred = np.array(
            [
                np.argmax(self.sess_log["out"]["pick"][i])
                for i in range(len(self.sess_log["out"]["pick"]))
            ]
        )
        y_true = np.array(self.sess_log["targets"]["action_instance"]).reshape(-1)
        return y_true, y_pred

    def get_category_tokens(self, instance_token_list: List) -> List:
        return [
            special_instance_vocab.index2word(idx).split(":")[0]
            for idx in instance_token_list
        ]

    def record(self, metrics: Dict, step: int = None) -> Dict:
        """prints the metrics;
        adds a confusion matrix if it
        TODO: print to file"""
        if step is not None:
            print(
                step,
                [
                    metrics[key]
                    for key in [
                        f"{self.name}/Loss",
                        f"{self.name}/Acc/symbol",
                        f"{self.name}/Acc/category_token",
                    ]
                ],
            )
        else:
            print(
                "final",
                [
                    metrics[key]
                    for key in [
                        f"{self.name}/Loss",
                        f"{self.name}/Acc/symbol",
                        f"{self.name}/Acc/category_token",
                    ]
                ],
            )
        return metrics

    def compute_confusion_matrix(self) -> wandb.plot.confusion_matrix:
        """Returns a wandb confusion matrix for current sess_log true and pred seq"""
        return wandb.plot.confusion_matrix(
            probs=None,
            y_true=self.sess_log["x_true"]["category_token"],
            preds=self.sess_log["x_pred"]["category_token"],
            class_names=list(category_dict.values()),
        )

    def rollout_learned_policy(self, session_id: int, savepath: str) -> Dict:
        """Env is initialized to startframe of session.
         learned policy are rolled out on the env
        Returns : Dict
        """
        obs = self.env.reset(ref_sess=self.session_paths[session_id])
        learned_policy = LearnedPolicy(
            self.model,
            self.config["context_history"],
            self.config["device"],
            self.config["pick_only"],
        )
        learned_policy.reset()
        learned_rollout_info = session_rollout(obs, self.env, learned_policy, savepath)
        return learned_rollout_info

    def compute_all_rollout_metrics(
        self, expert_rollout_info: Dict, learned_rollout_info: Dict
    ) -> Dict:
        """Calculates metrics for rollout of the learned policy wrt
        what an expert policy would have done in that scenario.
        """
        # Packing efficiency
        expert_placed_utensils = expert_rollout_info["placed_utensils"]
        learned_placed_utensils = learned_rollout_info["placed_utensils"]

        def get_packing_metrics(name: str) -> float:
            num_correct_category = self.env.env_utils.count_by_category_preference(
                learned_placed_utensils[name],
                getattr(self.env.env_utils.preference, f"category_order_{name}"),
            )
            return num_correct_category / len(expert_placed_utensils[name])

        self.main_log["TPE"].append(get_packing_metrics(name="top_rack"))
        self.main_log["BPE"].append(get_packing_metrics(name="bottom_rack"))
        # self.main_log["SPE"].append(get_packing_metrics(name="sink")
        self.main_log["PE"].append(
            (self.main_log["TPE"][-1] + self.main_log["BPE"][-1]) / 2.0
        )

        # Time efficiency
        expert_actions = expert_rollout_info["actions"]
        learned_actions = learned_rollout_info["actions"]
        success = self.main_log["PE"][-1]
        self.main_log["SPL"].append(
            success * len(expert_actions) / len(learned_actions)
        )

        # Levenshtein Distance for the pick category order
        expert_category_order = [
            action.pick_instance.category_name for action in expert_actions
        ]
        learned_category_order = [
            action.pick_instance.category_name for action in learned_actions
        ]
        self.main_log["LC"].append(
            textdistance.levenshtein(expert_category_order, learned_category_order)
            / max(len(expert_category_order), len(learned_category_order))
        )
        return

    def get_rollout_metrics(self, session_id: int, savepath: str) -> None:
        expert_rollout_info = self.expert_rollout_info[session_id]
        learned_rollout_info = self.rollout_learned_policy(session_id, savepath)
        self.compute_all_rollout_metrics(expert_rollout_info, learned_rollout_info)
        return
