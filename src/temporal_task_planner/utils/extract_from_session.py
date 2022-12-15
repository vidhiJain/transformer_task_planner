from glob import glob
from temporal_task_planner.constants.gen_sess_config.lookup import instance_dict

def get_placed_utensils(data):
    return data["session"]["placedUtensils"]


def get_rack_cam_coordinates(data):
    return data["session"]["rackCamera"]["translation"]


def get_start_keyframe(data, useraction):
    return data["session"]["keyframes"][useraction["startFrame"]]


def get_instance_to_track(useraction):
    if useraction["rigidObj"] == "":
        # ACT = dishwasher interaction
        track = "ktc_dishwasher_:0000_joint_" + str(useraction["articulatedLink"])
    else:
        track = useraction["rigidObj"]
    return track


def get_target_instance_poses(useraction):
    return {"initPose": useraction["initPose"], "endPose": useraction["endPose"]}


def get_feasible_picks(useraction):
    return [instance_dict[str(idx)] for idx in useraction["feasiblePick"]]


def get_feasible_place(useraction):
    # TODO compute with sim for current feasible locations in run on sim
    return useraction["feasiblePlace"]


def get_placed_utensils_in_useraction(data, useraction_idx=-1):
    return data["session"]["userActions"][useraction_idx]["placedUtensils"]