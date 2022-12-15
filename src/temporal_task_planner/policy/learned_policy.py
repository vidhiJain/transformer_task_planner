from typing import Any, Dict, List
from temporal_task_planner.data_structures.instance import RigidInstance
from temporal_task_planner.utils.gen_sess_utils import get_potential_placements
import torch
import os
from temporal_task_planner.constants.gen_sess_config.lookup import (
    special_instance_vocab,
    category_vocab,
)
from temporal_task_planner.data_structures.action import Action
from temporal_task_planner.data_structures.state import State
from temporal_task_planner.data_structures.temporal_context import TemporalContext
from temporal_task_planner.policy.abstract_policy import Policy
from temporal_task_planner.trainer.dataset import pad_fn
from temporal_task_planner.utils.place_pose_mgr import DishwasherPlacePoseManager
from temporal_task_planner.utils.data_structure_utils import (
    construct_placement_instance,
    construct_rigid_instance,
    get_dishwasher_part_pose,
    get_dishwasher_state,
    get_potential_place_instances_per_category,
    process_dishwasher_for_place_instances,
    get_picked_instance,
    get_place_instances,
)
from temporal_task_planner.constants.gen_sess_config.lookup import (
    original_potential_placements,
    dishwasher_part_poses,
    link_names_id,
)


class LearnedPolicy(Policy):
    def __init__(
        self,
        model: torch.nn.Module,
        context_history: int,
        device: str,
        pick_only: bool,
        chkpt_path: str = None,
    ) -> None:
        self.pick_only = pick_only
        self.chkpt_path = chkpt_path
        self.model = model
        self.context_history = context_history
        self.device = device
        bottom_rack_placements = get_potential_placements(
            original_potential_placements, is_top_rack=False
        )
        top_rack_placements = get_potential_placements(
            original_potential_placements, is_top_rack=True
        )
        self.potential_placements = {
            "top": top_rack_placements,
            "bottom": bottom_rack_placements,
        }

    def reset(self, **kwargs) -> None:
        """Does not make use of the env if passed in args"""
        self.counter = 0
        if self.chkpt_path is not None:
            self.init_model_state()
        self.history = TemporalContext()

    def init_model_state(self) -> None:
        assert os.path.exists(self.chkpt_path), "checkpoint does not exist~!"
        checkpoint = torch.load(self.chkpt_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def get_temporal_context(self, raw_inputs: State) -> TemporalContext:
        """Construct the (new) Temporal Context from State(s)
        depending on the context history budget set

        - No init action support needed as obs from sim
        - Timestep correction handled per state in the temporal context

        Reference:
        - From dataset.py process_session_keyframe fn
        - From transformer_task_planner repo trainer/eval/run_policy_on_sim.py get_temporal_context fn
        """
        temporal_context = self.history.init_sub_copy(
            max(0, self.counter - self.context_history), self.counter + 1
        )
        # apply relative timestep to instance of each state in temporal context
        for relative_timestamp, state in enumerate(temporal_context.states):
            state.apply_timestep(relative_timestamp)
        return temporal_context

    def process_model_input(
        self, temporal_context: TemporalContext
    ) -> Dict[str, torch.Tensor]:
        """Format the temporal context object into model input dict
        which contains all the instance attributes.
        Additionally, the src_mask and device for processing.
        """
        inputs = temporal_context.process_states()
        # Input Dict of attributes with torch tensors; unsqueeze for batch_size 1
        inputs_dict = {
            key: torch.tensor(inputs[key]).unsqueeze(0) for key in inputs.keys()
        }
        input_len = len(inputs_dict["timestep"][0])
        inputs_dict["src_key_padding_mask"] = torch.zeros(input_len).bool().unsqueeze(0)
        return inputs_dict

    def get_action(self, raw_inputs: State) -> Action:
        """Process logits into Action for env.step
        Args:
            raw_inputs: State (current frame)
        Returns:
            output: Action
        policy depends on temporal context (sequence of previous states)
        to predict the action, then construct it before passing to the model
        """
        raw_inputs.pick_only = True
        self.history.states.append(raw_inputs)
        temporal_context = self.get_temporal_context(raw_inputs)
        inputs = self.process_model_input(temporal_context)
        out = self.model(**inputs, device=self.device)
        # mask previous states' instances; leave only current rigid instances logits
        mask_len = len(inputs["timestep"][0])
        mask = [0.0] * mask_len
        for idx in range(0, mask_len - len(raw_inputs.rigid_instances) - 1):
            mask[idx] = -float("inf")
        # TODO: reword "pick" "place" dict in out
        pick_logits = out["pick"][-1] + torch.tensor(mask).to(self.device)
        # Taking the predicted index for the pick action!
        pred_idx = torch.argmax(pick_logits, dim=-1).reshape(-1).item()
        name = special_instance_vocab.index2word(inputs["instance_token"][0][pred_idx])
        pose = inputs["pose"][-1][pred_idx]
        raw_inputs.pick_only = False
        # Taking the predicted index for the place_instance
        picked_instance = get_picked_instance(name, pose)
        raw_inputs.place_instances = get_place_instances(picked_instance, self.potential_placements)
        # print(picked_instance)
        if self.pick_only:
            self.counter += 1
            return Action(pick_instance=picked_instance, place_instance=None)
        # else place prediction
        temporal_context = self.get_temporal_context(raw_inputs)
        inputs = self.process_model_input(temporal_context)
        out = self.model(**inputs, device=self.device)
        # mask the previous states and current rigid + pick act instance
        mask_len = len(inputs["timestep"][0])
        mask = [0.0] * mask_len
        for idx in range(0, mask_len - len(raw_inputs.place_instances) - 1):
            mask[idx] = -float("inf")
        place_logits = out["pick"][-1] + torch.tensor(mask).to(self.device)
        pred_idx = torch.argmax(place_logits, dim=-1).reshape(-1).item()
        place_instance = construct_placement_instance(
            picked_instance,
            inputs["pose"][0][pred_idx].cpu().detach().numpy().tolist(),
        )
        action = Action(
            pick_instance=picked_instance,
            place_instance=place_instance,
        )
        print(action)
        self.history.actions.append(action)
        self.counter += 1
        return action

    def update_history_record(self, obs: State, act: Action) -> None:
        self.history.states.append(obs)
        self.history.actions.append(act)
