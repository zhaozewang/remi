import numpy as np
from typing import Dict, Any, Optional, Union, List
from omegaconf import DictConfig


def parse_gc_profile(
        sensory_cfg: Dict[str, Any], 
        rep_seed: int, 
        room_seed: int
    ) -> Dict[str, Any]:
    """
    Parse the sensory profile for Grid Cell cells.

    **Parameters:**
        - `sensory_cfg` (dict): The sensory configuration.
        - `rep_seed` (int): The random seed for the sensory profile.
        - `room_seed` (int): The random seed for the room.

    **Returns:**
        - `gc_profile` (dict): The sensory profile for Grid Cell cells.
    """

    # Initialize the empty profile dict and RNG for a specific room
    gc_profile = {}
    ori_rng = np.random.RandomState(room_seed)

    # Constant scaling factor and the initial scale
    sr = np.sqrt(np.e)  # Constant scaling ratio
    l0 = sensory_cfg["initial_scale"]  # Initial scale

    def gc_profile_generator(scale: float, n_cells: int, rep_seed: int) -> Dict[str, Any]:
        use_old_gc = "theta" not in sensory_cfg
        return {
            "type": "grid_cell",
            "n_cells": n_cells,
            "scale": scale,
            "sigma2scale": sensory_cfg["sigma2scale"],
            "magnitude": sensory_cfg["magnitude"],
            "normalize": sensory_cfg["normalize"],
            "theta": sensory_cfg.get("theta", 60),
            "use_old_gc": use_old_gc,
            "seed": rep_seed,
        }

    # If multiple groups per scale, generate the profile for each group
    if sensory_cfg.get("n_groups_per_scale") is not None:
        n_cells = sensory_cfg["n_cells_per_group"]
        oris = ori_rng.uniform(0, 60, sensory_cfg["n_groups_per_scale"]).astype(np.int32)
        gc_profile.update({
            f"gc_{i}_{o}": {**gc_profile_generator(l0 * sr**i, n_cells, rep_seed), "orientation": oris[o]}
            for o in range(sensory_cfg["n_groups_per_scale"])
            for i in range(sensory_cfg["n_scales"])
        })

    # If only one group per scale, generate the profile for the group
    else:
        gc_profile.update({
            f"gc_{i}": gc_profile_generator(l0 * sr**i, sensory_cfg["n_cells_per_group"], rep_seed)
            for i in range(sensory_cfg["n_scales"])
        })

    return gc_profile


def parse_dir_profile(sensory_cfg: Dict[str, Any], rep_seed: int, *args, **kwargs) -> Dict[str, Any]:
    """
    Parse the sensory profile for Direction cells.

    **Parameters:**
        - `sensory_cfg` (dict): The sensory configuration.
        - `rep_seed` (int): The random seed for the sensory profile.

    **Returns:**
        - `dir_profile` (dict): The sensory profile for Direction cells.
    """
    sensory_cfg = dict(sensory_cfg)
    sensory_cfg["seed"] = rep_seed
    return { "dir": sensory_cfg }


def parse_vel_profile(sensory_cfg: Dict[str, Any], rep_seed: int, *args, **kwargs) -> Dict[str, Any]:
    """
    Parse the sensory profile for Velocity cells.

    **Parameters:**
        - `sensory_cfg` (dict): The sensory configuration.
        - `rep_seed` (int): The random seed for the sensory profile.

    **Returns:**
        - `vel_profile` (dict): The sensory profile for Velocity cells.
    """
    sensory_cfg = dict(sensory_cfg)
    sensory_cfg["seed"] = rep_seed
    return { "vel": sensory_cfg }


def parse_wsm_profile(sensory_cfg: Dict[str, Any], rep_seed: int, room_seed: int) -> Dict[str, Any]:
    """
    Parse the sensory profile for Weak Spatially Modulated (WSM) cells.

    **Parameters:**
        - `sensory_cfg` (dict): The sensory configuration.
        - `rep_seed` (int): The random seed for the sensory profile.
        - `room_seed` (int): The random seed for the room.

    **Returns:**
        - `wsm_profile` (dict): The sensory profile for Weak Spatially Modulated (WSM) cells.
    """
    wsm_profile = {}
    wsm_profile[f"wsm"] = {
        "type": "weak_sm_cell",
        "n_cells": sensory_cfg["n_cells"],
        "sigma": sensory_cfg["sigma"],
        "ssigma": sensory_cfg["ssigma"],
        "magnitude": sensory_cfg["magnitude"],
        "normalize": sensory_cfg["normalize"],
        "seed": rep_seed + room_seed
    }
    return wsm_profile


def parse_single_sensory_cfg(
        sensory_cfg: DictConfig[str, Any], 
        rep_seed: int, 
        room_seed: int
    ) -> Dict[str, Any]:
    """
    Parse the sensory profile from a single sensory configuration.

    **Parameters:**
        - `sensory_cfg` (dict): The sensory configuration.
        - `rep_seed` (int): The random seed for the sensory profile.
        - `room_seed` (int): The random seed for the room.

    **Returns:**
        - `sensory_profile` (dict): The sensory profile from a single sensory configuration.
    """
    parser_dict = {
        "grid_cell": parse_gc_profile,
        "weak_sm_cell": parse_wsm_profile,
        "direction_cell": parse_dir_profile,
        "velocity_abs": parse_vel_profile
    }
    if sensory_cfg.type not in parser_dict:
        raise ValueError(f"Invalid key for sensory type: {sensory_cfg.type}")
    return parser_dict[sensory_cfg.type](
        sensory_cfg=sensory_cfg, 
        rep_seed=rep_seed, 
        room_seed=room_seed
    )


def parse_sensory_cfg(
        rep_seed: int, 
        room_seed: int,
        sensory_cfg_list: Optional[List[Dict[str, Any]]] = None, 
    ) -> Dict[str, Any]:
    """
    Parse the sensory profile from the sensory configuration list.

    **Parameters:**
        - `rep_seed` (int): The random seed for the sensory profile.
        - `room_seed` (int): The random seed for the room.
        - `sensory_cfg_list` (list): The sensory configuration list.

    **Returns:**
        - `sensory_profile` (dict): The sensory profile from the sensory configuration list.
    """
    sensory_profile = {}
    if sensory_cfg_list is not None:
        for sensory_cfg in sensory_cfg_list.values():
            _single_sensory_profile = parse_single_sensory_cfg(
                sensory_cfg, rep_seed, room_seed
            )
            sensory_profile.update(_single_sensory_profile)
    return sensory_profile


def count_cell_num(sensory_profile: Dict[str, Any], keys: Optional[List[str]] = None) -> int:
    """
    Count the number of cells in the sensory profile.

    **Parameters:**
        - `sensory_profile` (dict): The sensory profile.
        - `keys` (list): The keys to count.

    **Returns:**
        - `n_cells` (int): The number of cells in the sensory profile.
    """
    if keys is None:
        return sum(v["n_cells"] for v in sensory_profile.values())
    else:
        return sum(sensory_profile[key]["n_cells"] for key in keys)


def filter_sensory_profile(
        sensory_profile: Dict[str, Any], 
        keys: Optional[Union[str, List[str]]] = None, 
        type_filter: Optional[str] = None, 
        str_filter: Optional[str] = None
    ) -> List[str]:
    """
    Filter the sensory profile based on the keys, type, or string filter.

    **Parameters**
        - `sensory_profile` (dict): The sensory profile to filter.
        - `keys` (list): The keys to filter.
        - `type_filter` (str): The type to filter.
        - `str_filter` (str): The string to filter.

    **Returns:**
        - `filtered_keys` (list): The keys of the sensory profile that match the filter.
    """
    if keys is not None:
        if isinstance(keys, str):
            keys = [keys]
        filtered_keys = list(sensory_profile.keys())
    elif type_filter is not None:
        filtered_keys = [k for k, v in sensory_profile.items() if v["type"] == type_filter]
    elif str_filter is not None:
        filtered_keys = [k for k in sensory_profile.keys() if str_filter in k]
    return filtered_keys


def filter_profile_by_strings(
        sensory_profile: Dict[str, Any], 
        str_filter_list: Optional[List[str]] = None
    ) -> List[str]:
    """
    Filter the sensory profile based on the string filter list.

    **Parameters:**
        - `sensory_profile` (dict): The sensory profile to filter.
        - `str_filter_list` (list): The list of strings to filter.

    **Returns:**
        - `filtered_keys` (list): The keys of the sensory profile that match the filter.
    """
    all_keys = list(sensory_profile.keys())
    filtered_keys = []
    for str_filter in str_filter_list:
        filtered_keys.extend([k for k in all_keys if str_filter in k])
    filtered_keys = sorted(list(set(filtered_keys)))
    return filtered_keys
