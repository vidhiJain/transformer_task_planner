from habitat_sim import SensorType, SensorSubType

height = 320
width = 420
min_point_occ_thresh = 0
visibility_count_thresh = 0.0  
pick_accesible_count_thresh = 0.75

cam_info = [
    {
        "uuid": "color_kitchen_sensor",
        "type": SensorType.COLOR,
        "subtype": SensorSubType.ORTHOGRAPHIC,
    },
    {
        "uuid": "depth_kitchen_sensor",
        "type": SensorType.DEPTH,
        "subtype": SensorSubType.ORTHOGRAPHIC,
    },
    {
        "uuid": "semantic_kitchen_sensor",
        "type": SensorType.SEMANTIC,
        "subtype": SensorSubType.ORTHOGRAPHIC,
    },
    {
        "uuid": "color_rack_sensor",
        "type": SensorType.COLOR,
        "subtype": SensorSubType.ORTHOGRAPHIC,
    },
    {
        "uuid": "depth_rack_sensor",
        "type": SensorType.DEPTH,
        "subtype": SensorSubType.ORTHOGRAPHIC,
    },
    {
        "uuid": "semantic_rack_sensor",
        "type": SensorType.SEMANTIC,
        "subtype": SensorSubType.ORTHOGRAPHIC,
    },
]

default_cam = {
    "translation": [-1.059, 1.0, 0.899],
    "rotation": [0.9423165, -0.734723, 0.0, 0.0],
}

kitchen_cam = {
    "translation": [-1.59, 1.10, 0.946],
    "rotation": [0.9423165, -0.734723, 0.0, 0.0],
    "zoom": 5,
}

rack_cam = {
    "translation": [-1.059, 1.0, 0.95],
    "rotation": [0.9423165, -0.734723, 0.0, 0.0],
    "zoom": 18,
}