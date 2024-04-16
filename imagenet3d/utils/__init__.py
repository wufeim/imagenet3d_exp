from .configuration import load_config, save_config
from .distributed_utils import is_main_process, is_dist_enabled, get_world_size, get_global_rank
from .dnnlib import EasyDict, construct_class_by_name, call_func_by_name
from .general import setup_logging
from .meter import AverageMeter
from .multi_bin import continuous_to_bin, bin_to_continuous, MultiBin
from .pose_utils import pose_error

IMAGENET_CATES = [
    'aeroplane', 'airship', 'ambulance', 'ashtray', 'ax', 'backpack', 'basket', 'beaker', 'bed', 'bench',
    'bicycle', 'bicycle_built_for_two', 'blackboard', 'boat', 'bookshelf', 'bottle', 'bow', 'bowl', 'bucket', 'bus',
    'cabinet', 'calculator', 'camera', 'can', 'cap', 'car', 'cart', 'cellphone', 'chain_saw', 'chair',
    'clock', 'coffee_maker', 'comb', 'computer', 'crutch', 'cup', 'diningtable', 'dishwasher', 'door', 'dumbbell',
    'eraser', 'eyeglasses', 'fan', 'faucet', 'filing_cabinet', 'fire_extinguisher', 'fire_truck', 'fish_tank', 'flashlight', 'fork',
    'forklift', 'french_horn', 'garbage_truck', 'go_kart', 'guitar', 'hair_dryer', 'hammer', 'hand_barrow', 'harp', 'harvester',
    'headphone', 'helmet', 'hourglass', 'iron', 'jar', 'jinrikisha', 'kettle', 'key', 'keyboard', 'knife',
    'lighter', 'mailbox', 'microphone', 'microwave', 'motorbike', 'mouse', 'padlock', 'paintbrush', 'pan', 'pay_phone',
    'pen', 'pencil', 'piano', 'pillow', 'plate', 'pot', 'power_drill', 'printer', 'punching_bag', 'racket',
    'recreational_vehicle', 'refrigerator', 'remote_control', 'rifle', 'road_pole', 'salt_or_pepper_shaker', 'satellite_dish', 'sax', 'screwdriver', 'shoe',
    'shovel', 'sink', 'skate', 'skateboard', 'slipper', 'snowmobile', 'sofa', 'speaker', 'spoon', 'stapler',
    'stove', 'suitcase', 'teapot', 'telephone', 'toaster', 'toilet', 'toothbrush', 'train', 'trash_bin', 'trophy',
    'tvmonitor', 'vending_machine', 'violin', 'washing_machine', 'watch', 'wheelchair']
