from copy import deepcopy
from glob import glob
import json
import os
from pathlib import Path
import random
from typing import Any, Dict
import gym
import numpy as np
from omegaconf import DictConfig
import wandb
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from temporal_task_planner.trainer.dataset import (
    DishwasherArrangeDataset,
    DishwasherArrangeSavedSession,
    pad_fn,
)
from temporal_task_planner.trainer.early_stopping import EarlyStopping
from temporal_task_planner.trainer.logger import Logger
from temporal_task_planner.utils.datasetpytorch_utils import get_session_path, get_session_paths_split


def construct_run_name(cfg):
    """Custom name format for comparison plots"""
    return f"cw-{cfg['config']['context_history']}_b-{cfg['config']['batch_size']}"  # nT-{cfg['config']['num_targets_per_step']}"


class Learner:
    """Learner class that provides training support for
    any Transformer Task Planner model
    NOTE:
    - wandb calls are in `train` and `train_batch`
    - hydra recursively instantiates all modules in learner
    - optimizer and scheduler are partial modules as they're dependent on other modules
        (namely model.parameters(), optimizer respectively)
    """

    def __init__(
        self,
        config: DictConfig,
        model: PreTrainedModel,
        criterion: torch.nn.Module,
        optimizer_partial: Any,  # ModuleWrapper for torch.optim
        scheduler_partial: Any,  # ModuleWrapper for torch.optim.lr_scheduler
        env: Any, # gym.Env,  # for rollouts
    ) -> None:
        # fixed while training
        self.config = config
        self.session_paths = {}
        self.session_paths = get_session_paths_split(Path(config['pkg_root'] , config['dataset_path']).as_posix())
        # for name in ["train", "val", "test"]:
        #     self.session_paths[name] = get_session_path(
        #         Path(
        #             self.config.pkg_root, self.config.dataset_path, name, "session_json"
        #         ).as_posix()
        #     )
        self.chkpt_path = (
            f"{self.config['chkpt_name']}{self.config['context_history']}.pt"
        )
        self.env = env
        self.fix_seed()
        if self.config["batch_size"] == 0:
            self.set_adaptative_batch()
        else:
            self.batch_size = self.config["batch_size"]
        assert self.batch_size is not None or self.batch_size != 0
        self.train_loader = self.get_dataloader("train", self.batch_size, shuffle=False)
        self.max_train_evals = int(config["max_train_evals"])
        self.max_val_evals = int(config["max_val_evals"])

        # changes while training
        self.model = model
        self.model.to(device=self.config["device"])
        self.criterion = criterion
        self.optimizer = optimizer_partial(params=model.parameters())
        self.scheduler = scheduler_partial(optimizer=self.optimizer)
        self.early_stopping = EarlyStopping(
            patience=self.config["patience"],
            verbose=True,
            path=self.chkpt_path,
        )
        self.init_all_loggers()

    def fix_seed(self):
        seed = self.config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
        return

    def set_adaptative_batch(self):
        self.batch_size = self.config["num_targets_per_step"] // (
            self.config["context_history"] + 1
        )

    def init_all_loggers(self) -> None:
        self.train_logger = Logger(
            self.env,
            self.session_paths["train"],
            self.config,
            self.model,
            self.criterion,
            name="Train",
            pick_only=self.config['pick_only'],
            max_evals=self.max_train_evals,
        )
        self.val_logger = Logger(
            self.env,
            self.session_paths["val"],
            self.config,
            self.model,
            self.criterion,
            name="Val",
            pick_only=self.config['pick_only'],
            max_evals=self.max_val_evals,
        )

    def load_chkpt(self) -> None:
        assert os.path.exists(self.chkpt_path), "checkpoint does not exist~!"
        checkpoint = torch.load(self.chkpt_path)  # wandb.restore(self.chkpt_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.loss = checkpoint["loss"]
        self.step = checkpoint["step"]
        self.example_ct = checkpoint["example_ct"]
        print(f"Loaded model={self.model}, optimizer={self.optimizer}")
        print(f"epoch={self.epoch}, step={self.step}, example_ct={self.example_ct}")
        return

    def init_training(self) -> None:
        if self.config["load_chkpt"] and os.path.exists(self.chkpt_path):
            print("Loading checkpoint!")
            self.load_chkpt()
        else:
            self.epoch = 0
            self.step = 0
            self.example_ct = 0
            print(
                f"No checkpoint found. Starting with epoch={self.epoch}, step={self.step}, example_ct={self.example_ct}"
            )
        return

    def init_dataset(self, dataset_type="train") -> DishwasherArrangeDataset:
        data = DishwasherArrangeDataset(
            session_paths=self.session_paths[dataset_type],
            context_history=self.config["context_history"],
            num_sessions_limit=self.config["max_session_data_limit"],
            pick_only=self.config["pick_only"],
        )
        return data

    def get_dataloader(
        self, dataset_type: str = "train", batch_size=1, shuffle=False
    ) -> DataLoader:
        data = self.init_dataset(dataset_type)
        print(f"{dataset_type} data sequences: {len(data)}")
        loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=pad_fn,
        )
        return loader

    def log_metrics(self, to_rollout: bool) -> None:
        print(f"to_rollout={to_rollout}")
        metrics = {}
        metrics.update(self.train_logger(self.model, self.step, to_rollout))
        metrics.update(self.val_logger(self.model, self.step, to_rollout))
        metrics.update({"LearningRate": self.scheduler.get_last_lr()[0]})
        metrics.update({"ExampleCounts": self.example_ct})
        print(json.dumps(metrics, indent=4))
        wandb.log(metrics, step=self.step)

    def train_batch(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        input, target = data
        out = self.model(**input, device=self.config["device"])
        assert (
            out["pick"].shape[0] == torch.cat(target["action_instance"], dim=0).shape[0]
        ), "out and target size do not match"
        loss = self.criterion(out, target)
        if self.step % self.config["logging_interval"] == 0:
            wandb.log({'RealTrainLoss' : loss.item()}, self.step)
        loss.backward()
        self.optimizer.step()
        
        to_rollout = True if self.step % self.config["rollout_interval"] == 0 else False
        if self.step % self.config["logging_interval"] == 0:
            self.log_metrics(to_rollout)
            self.scheduler.step()
            # data = self.val_logger.log_true_pred_tokens(self.step)
            # self.wandb_table_true_pred.add_data(*data)
            val_loss = np.mean(self.val_logger.main_log["Loss"])
            save_dict = {
                "example_ct": self.example_ct + input["action_masks"].sum().item(),
                "step": self.step + 1,
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "loss": loss,
            }
            saved_checkpoint = self.early_stopping(
                val_loss,
                save_dict=save_dict,
            )
            torch.save(save_dict, self.chkpt_path)
            wandb.save(self.chkpt_path)

            if self.early_stopping.early_stop:
                print("Early stopping")
                # wandb.log({"Log": self.wandb_table_true_pred})
                return self.model.state_dict()


        self.step += 1
        self.example_ct += input["action_masks"].sum().item()
        return

    def train(self):
        self.init_training()
        while self.step <= self.config["max_steps"]:
            for data in self.train_loader:
                self.train_batch(data)
            self.epoch += 1
        # wandb.log({"Sample True Pred": self.wandb_table_true_pred})
        return self.model.state_dict()

    # def init_wandb(self, cfg: Dict) -> Any:
    #     self.wandb_run = wandb.init(
    #         project=f"learner_debug",
    #         entity="dishwasher_arrange",
    #         config=cfg,
    #         save_code=False,
    #         name=construct_run_name(cfg),
    #     )


if __name__ == "__main__":
    import hydra

    @hydra.main(config_path="../../../config", config_name="learner")
    def main(cfg):
        cfg.pkg_root = hydra.utils.get_original_cwd()
        learner = hydra.utils.instantiate(cfg.learner)
        learner.init_wandb(cfg)
        with wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=cfg,
            save_code=False,
            name=construct_run_name(cfg),
            resume=True,
        ) as run:
            learner.train()

    main()
