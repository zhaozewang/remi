import torch
from omegaconf import OmegaConf
from typing import Dict, Any, Union, Optional

import remi.utils.sensory_profile as sensory_profile
import remi.utils.loss_fn as loss_fn
import remi.rtgym_lite as rtgym
from remi.rtgym_lite.dataclass import Trajectory, AgentState
from remi.model.masking import Masking
from remi.nn4n.nn import RecurrentLayer, LeakyLinearLayer, LinearLayer


class HCMECBlock(RecurrentLayer):
    def __init__(self, region_cfg: Dict[str, Any], rep_seed: int, room_idx: int, device: str = "cpu"):
        super().__init__()
        # Set device
        self.device = device

        # Initialize _masking_fn as None
        self._masking_fn = None
        self._masking_prob = None

        # Parse region cfg
        self.region_cfg = region_cfg
        self._construct_sensory_profile(rep_seed, room_idx)
        self._construct_io_profile()

        # Construct input mask (if specified)
        self._construct_input_mask()
        self._offset = self.region_cfg.get("offset", 0)

        # Construct projection layer and leaky layer
        self._construct_projection_layer()
        self._construct_leaky_layer()

        self.is_test = False

    # ------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------
    @property
    def loss_cfg(self):
        return self.region_cfg.get("loss_cfg", {})

    @property
    def use_masking(self):
        return self._masking_fn is not None

    @property
    def masking_fn(self):
        # If in test mode or masking prob is 0, don't mask (return identity function)
        if self.is_test or self.masking_prob <= 0 or self._masking_fn is None:
            return lambda x: x
    
        # Else, return the masking function
        else:
            def _masking_fn(data):
                if self.masking_method == "random":
                    mr = torch.rand(data.shape[0], device=self.device) * self.masking_prob
                elif self.masking_method == "constant":
                    mr = torch.ones(data.shape[0], device=self.device) * self.masking_prob
                else:
                    raise ValueError(f"Invalid masking method: {self.masking_method}")
                return self._masking_fn(data, mr)
            return _masking_fn

    @masking_fn.setter
    def masking_fn(self, value):
        self._masking_fn = value

    @property
    def masking_prob(self):
        if self._masking_prob is None:
            self._masking_prob = self.region_cfg.get("masking_prob", 0)
        return self._masking_prob

    @masking_prob.setter
    def masking_prob(self, value):
        self._masking_prob = value

    def to(self, device):
        super().to(device)
        self.device = torch.device(device) if isinstance(device, str) else device
        if self._masking_fn is not None:
            self._masking_fn.to(self.device)

    def eval(self, *args, **kwargs):
        super().eval(*args, **kwargs)
        self.is_test = True

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.is_test = False

    # ------------------------------------------------------------
    # Sensory Profile and Parsing
    # ------------------------------------------------------------
    def compute_pads(self, min_offset, max_offset):
        self._front_pad = max_offset - self._offset
        self._back_pad = min_offset - self._offset

    def sensory_keys(self, part="label"):
        assert part in ["label", "input", "all"]
        if self.sensory_profile != {}:
            return sensory_profile.filter_sensory_profile(
                self.sensory_profile,
                str_filter=self.region_cfg.data_cfg[part]["keys"]
            )
        else:
            return []

    def set_room(self, rep_seed: int, room_idx: int):
        self._construct_sensory_profile(rep_seed, room_idx)

    def _construct_sensory_profile(self, rep_seed: int, room_idx: int):
        """
        Construct the sensory profile from the region_cfg.
        If sensory profile is not provided, set it to an empty dictionary. That is,
        the region will not correspond to any simulation, but the neurons will be
        modeled.
        """
        # If sensory profile is provided, parse it
        if "sensory_profile" in self.region_cfg:
            self.sensory_profile = sensory_profile.parse_sensory_cfg(
                rep_seed, room_idx, self.region_cfg.sensory_profile
            )
        else:
            self.sensory_profile = {}

    def _construct_io_profile(self):
        """
        Construct the input and label keys from the region_cfg.
        If input or label keys are not provided, set them to an empty list.
        """
        # If data_cfg is provided, construct the input and label keys
        if "data_cfg" in self.region_cfg:
            if "input" in self.region_cfg.data_cfg:
                input_filters = self.region_cfg.data_cfg.input["keys"]
                self.input_keys = sensory_profile.filter_profile_by_strings(
                    sensory_profile=self.sensory_profile, 
                    str_filter_list=input_filters
                )
            else:
                self.input_keys = []
            
            if "label" in self.region_cfg.data_cfg:
                label_filters = self.region_cfg.data_cfg.label["keys"]
                self.label_keys = sensory_profile.filter_profile_by_strings(
                    sensory_profile=self.sensory_profile, 
                    str_filter_list=label_filters
                )
            else:
                self.label_keys = []

            self.input_dimension = sensory_profile.count_cell_num(self.sensory_profile, self.input_keys)
            self.hidden_dimension = sensory_profile.count_cell_num(self.sensory_profile, self.label_keys)

        # If data_cfg is not provided, set the input and label keys to empty lists
        else:
            self.input_keys = []
            self.label_keys = []
            self.input_dimension = 0 if "input_dim" not in self.region_cfg else self.region_cfg.input_dim
            self.hidden_dimension = self.region_cfg.hidden_dim  # Must be provided

    def _construct_input_mask(self):
        """
        Check if both masking_prob and masking_method are provided in the region_cfg.
        If so, construct the masking function. Otherwise, set the masking function to None.
        """
        self.masking_prob = OmegaConf.select(self.region_cfg, "masking_prob", default=0)
        self.masking_method = OmegaConf.select(self.region_cfg, "masking_method", default=None)
        if self.masking_prob > 0 and self.masking_method in ["random", "constant"]:
            self._masking_fn = Masking(device=self.device)
        else:
            self._masking_fn = None

    # ------------------------------------------------------------
    # Model Construction
    # ------------------------------------------------------------
    def _construct_projection_layer(self):
        if self.input_dimension > 0:
            assert self.input_dimension == self.hidden_dimension, "For HCMEC block, input and hidden dimensions must be the same"
            sparsity_mask = torch.eye(self.input_dimension)
            projection_layer = LinearLayer(
                input_dim=self.input_dimension, 
                output_dim=self.hidden_dimension,
                sparsity_mask=sparsity_mask
            )
            projection_layer.weight.data = torch.eye(self.input_dimension)
        else:
            projection_layer = None
        
        self.projection_layer = projection_layer

    def _construct_leaky_layer(self):
        if self.region_cfg["weight"] == "identity":
            diag_mask = torch.eye(self.hidden_dimension)
            linear_layer = LinearLayer(
                input_dim=self.hidden_dimension, 
                output_dim=self.hidden_dimension, 
                sparsity_mask=diag_mask
            )
            linear_layer.weight.data = torch.eye(self.hidden_dimension)
        elif self.region_cfg["weight"] == "zeros":
            linear_layer = LinearLayer(
                input_dim=self.hidden_dimension, 
                output_dim=self.hidden_dimension,
                sparsity_mask=torch.zeros(self.hidden_dimension, self.hidden_dimension)
            )
            linear_layer.weight.data = torch.zeros(self.hidden_dimension, self.hidden_dimension)
        else:
            linear_layer = LinearLayer(
                input_dim=self.hidden_dimension, 
                output_dim=self.hidden_dimension
            )

        leaky_layer = LeakyLinearLayer(
            linear_layer=linear_layer,
            activation=torch.nn.ReLU(),
            alpha=self.region_cfg["alpha"],
            learn_alpha=self.region_cfg["learn_alpha"]
        )
        self.leaky_layer = leaky_layer

    # ------------------------------------------------------------
    # Data Generation
    # ------------------------------------------------------------
    def trim_data(self, data: torch.Tensor, keep_front: bool = True):
        if data is None:
            return None
        _fp = self._front_pad + 1 if not keep_front else self._front_pad
        return data[:, _fp:None if self._back_pad == 0 else self._back_pad]

    def _generate_input_label(self, gym: rtgym.RatatouGym, traj: Trajectory):
        # Initialize variables
        input_res, label_res = None, None

        # Generate label response
        if len(self.label_keys) > 0:
            label_res = gym.agent.get_response(keys=self.label_keys, agent_data=traj)
            label_res = torch.tensor(label_res, dtype=torch.float32, device=self.device)
            # print(label_res.mean().item(), label_res.max().item(), label_res.min().item())

        # Generate input response
        if len(self.input_keys) > 0:
            input_res = gym.agent.get_response(keys=self.input_keys, agent_data=traj)
            input_res = torch.tensor(input_res, dtype=torch.float32, device=self.device)
            if hasattr(self.region_cfg, "noise") and self.region_cfg.noise > 0:
                input_std = input_res.std(dim=(0, 1), keepdim=True)
                input_res = input_res + torch.randn_like(input_res) * self.region_cfg.noise * input_std

        return input_res, label_res

    def generate_data(
            self, 
            gym: rtgym.RatatouGym,
            traj: Optional[Trajectory]=None, 
            override: Optional[Dict[str, torch.Tensor]]=None
        ):
        # ===== Generate untrimmed input and label response =====
        if override is not None:
            # Use override data
            input_res, label_res = override["input"].clone(), override["label"].clone()
        else:
            # Generate input and label response
            input_res, label_res = self._generate_input_label(gym, traj)

        if self.use_masking and input_res is not None:
            input_res = self.masking_fn(input_res)

        # ===== Generate initial state =====
        data_cfg = OmegaConf.select(self.region_cfg, "data_cfg", default=None)
        init_method = OmegaConf.select(self.region_cfg, "data_cfg.init", default=None)
        if data_cfg is None or init_method == "zeros_init":
            batch_size = traj.coord.shape[0] if traj is not None else override.shape[0]
            init_state = torch.zeros((batch_size, self.hidden_dimension))
            init_state = init_state.to(self.device)
        elif init_method == "label_init":
            assert label_res is not None
            init_state = label_res[:, 0]
        elif init_method == "input_init":
            assert input_res is not None
            init_state = input_res[:, 0]
        elif init_method == "masked_init":
            if label_res is not None and self.use_masking:
                init_state = self.masking_fn(label_res)[:, 0]
            else:
                init_state = label_res[:, 0]
        else:
            raise ValueError(f"Invalid init method: {init_method}")

        # ===== Trim input and label response =====
        input_res = self.trim_data(input_res, keep_front=False)
        label_res = self.trim_data(label_res, keep_front=False)

        return init_state, input_res, label_res
    
    def sample_data(self, gym: rtgym.RatatouGym, agent_data: Union[Trajectory, AgentState]):
        # Initialize variables
        label_keys = None

        # Generate input response
        if len(self.label_keys) > 0:
            label_keys = gym.agent.get_response(keys=self.label_keys, agent_data=agent_data)
            label_keys = torch.tensor(label_keys, dtype=torch.float32, device=self.device)

        return label_keys

    # ------------------------------------------------------------
    # Loss Computation
    # ------------------------------------------------------------
    def compute_loss(self, state: torch.Tensor, label: torch.Tensor, warmup_ts: int = 0):
        log_dict = {
            "mse_norm": loss_fn.mse_norm_loss(state, label, warmup_ts),
            "mse": loss_fn.mse_loss(state, label, warmup_ts),
            "fr_mse": loss_fn.fr_mse_loss(state, label, warmup_ts)
        }
        loss_dict = {}
        for loss_key, loss_lambda in self.loss_cfg.items():
            if loss_key == "mse_norm":
                loss_dict[loss_key] = log_dict[loss_key] * loss_lambda
            elif loss_key == "mse":
                loss_dict[loss_key] = log_dict[loss_key] * loss_lambda
            elif loss_key == "fr_mse":
                loss_dict[loss_key] = log_dict[loss_key] * loss_lambda
            else:
                raise ValueError(f"Invalid loss key: {loss_key}")
        return sum(loss_dict.values()), loss_dict, log_dict
