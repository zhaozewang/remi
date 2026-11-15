"""
A RNN model of Hippocampus and Medial Entorhinal Cortex (MEC). This model file provides a stand-alone implementation of the MEC model.
It keeps track of all the layers and the parameters.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

import remi.nn4n as nn4n
import remi.rtgym_lite as rtgym
import remi.utils as utils
import remi.utils.rnn_utils as rnn_utils
from remi.model.hcmec_block import HCMECBlock
from remi.nn4n.nn import RNN, BlockRecurrentLayer, TensorPack


class HCMEC(nn.Module):
    def __init__(
            self, hcmec_cfg: Dict[str, Any], rep_seed: int, room_idx: int,
            device: str = "cpu", postact_noise: float = 0.0, **kwargs
        ):
        super().__init__()
        self.region_dict = {}
        self.model = None
        self.hcmec_cfg = hcmec_cfg
        self.device = torch.device("cpu")
        self.rep_seed = rep_seed
        self.room_idx = room_idx
        self.postact_noise = postact_noise

        self._initialize_model()
        self._compute_pads()

    # ------------------------------------------------------------
    # Properties and Common Utils
    # ------------------------------------------------------------    
    @property
    def n_blocks(self):
        return self.model.recurrent_layers[0].n_blocks

    @property
    def sensory_profile(self):
        return self.aggregate_sensory_profiles()
    
    @property
    def behavior_profile(self):
        return self.hcmec_cfg["behavior"]
    
    @property
    def size(self):
        assert hasattr(self, "model"), "HCMEC model has not been set yet"
        return self.model.recurrent_layers[0].hidden_size
    
    @property
    def rec(self):
        assert hasattr(self, "model"), "HCMEC model has not been set yet"
        return self.model.recurrent_layers[0]
    
    def eval(self):
        for region_key in self.region_dict.keys():
            region = self.get_region(region_key)
            region.eval()

    def train(self):
        for region_key in self.region_dict.keys():
            region = self.get_region(region_key)
            region.train()

    def freeze(self):
        self.model.eval()
        self.model.freeze()

    def unfreeze(self):
        self.model.train()
        self.model.unfreeze()

    def region_indices(self, region_key):
        region_idx = self.region_dict[region_key]
        return self.model.recurrent_layers[0].block_indices(region_idx)

    def region_name(self, region_idx):
        return list(self.region_dict.keys())[region_idx]

    def region_dim(self, region_key):
        return self.rec.get_recurrent(self.region_dict[region_key]).hidden_size

    # ------------------------------------------------------------
    # Common Utils
    # ------------------------------------------------------------
    def to(self, device):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)
        for region_key in self.region_dict.keys():
            region = self.get_region(region_key)
            region.to(self.device)

    def eval(self):
        self.model.eval()
        for region_key in self.region_dict.keys():
            region = self.get_region(region_key)
            region.eval()

    def train(self):
        self.model.train()
        for region_key in self.region_dict.keys():
            region = self.get_region(region_key)
            region.train()

    # ------------------------------------------------------------
    # Data Generation
    # ------------------------------------------------------------
    def set_gym(self, gym_cfg: Dict[str, Any]):
        # Set up the gym
        self.gym_cfg = gym_cfg
        utils.printc("Setting up the gym", 'g')
        self.gym = rtgym.RatatouGym(
            temporal_resolution=gym_cfg["temporal_resolution"],
            spatial_resolution=gym_cfg["spatial_resolution"]
        )
        self.gym.init_arena_map(
            shape=gym_cfg["arena_cfg"]["arena_shape"], 
            **gym_cfg["arena_cfg"]["arena_params"]
        )
        self.gym.agent.set_behavior(self.behavior_profile)
        self.gym.agent.set_sensory(self.sensory_profile)

    def set_room(self, rep_seed: int, room_idx: int):
        if rep_seed == self.rep_seed and room_idx == self.room_idx:
            return
        self.rep_seed = rep_seed
        self.room_idx = room_idx
        utils.printc("Switching room", 'g')
        for region_key in self.region_dict.keys():
            region = self.get_region(region_key)
            region.set_room(rep_seed, room_idx)
        self.gym.agent.set_sensory(self.sensory_profile)

    def generate_data(self, duration, batch_size, traj=None):
        if traj is None:
            # Start a new trial and get the trial data
            traj = self.gym.agent.random_traverse(
                duration=duration, 
                batch_size=batch_size
            )
        n = len(self.region_dict)
        input_tp, label_tp, init_tp = nn4n.empty_tp(n), nn4n.empty_tp(n), nn4n.empty_tp(n)

        for region_key, region_idx in self.region_dict.items():
            generated_data = self.get_region(region_key).generate_data(
                gym=self.gym, 
                traj=traj
            )
            init_tp[region_idx] = generated_data[0]
            input_tp[region_idx] = generated_data[1]
            label_tp[region_idx] = generated_data[2]
            # if label_tp[region_idx] is not None:
            #     lb = label_tp[region_idx]
            #     print(region_key, lb.mean(), lb.std(), lb.min(), lb.max())

        # Init tensor pack need to be 2D because it its shape is (n_recurrent_layers, n_regions)
        # But since we only have one recurrent layer, just wrap it again with a TensorPack
        init_tp = TensorPack([init_tp])

        # Also return the coordinates
        coords = torch.tensor(traj.float_coord, device=self.device, dtype=torch.float32)
        coords = coords[:, self.default_pads[0]+1:None if self.default_pads[1] == 0 else self.default_pads[1]]

        # Return the data
        return init_tp, input_tp, label_tp, coords

    def generate_query_data(
            self, 
            query: torch.Tensor, 
            duration: float
        ):
        """
        Generate the sensory cue for the query region

        **Parameters:**
            - query: (batch_size, n_dim) The sensory cue to query the model
            - duration: float, in seconds

        **Returns:**
            - init_tp: (batch_size, n_dim)
            - input_tp: (batch_size, n_dim)
        """
        assert query.shape[1] == len(self.region_indices("wsm"))

        # This is just to get some random data so that we could set the size of the input tensor
        sample_traj = self.gym.agent.random_traverse(
            duration=duration, 
            batch_size=query.shape[0]
        )

        input_tp = nn4n.empty_tp(len(self.region_dict))
        init_tp = nn4n.empty_tp(len(self.region_dict))
        for region_key, region_idx in self.region_dict.items():
            sample_data = self.get_region(region_key).generate_data(self.gym, sample_traj)
            if region_key == "wsm":
                region = self.get_region(region_key)
                query = query.unsqueeze(1).repeat(1, self.gym.to_ts(duration), 1)
                if region.masking_fn is not None and region.masking_prob > 0:
                    query = region.masking_fn(query)
                query = region.trim_data(query, keep_front=False)
                input_tp[region_idx] = query
            else:
                input_tp[region_idx] = sample_data[1]
                if input_tp[region_idx] is not None:
                    # If the input is not None (i.e., network expects input)
                    # set it to zeros
                    input_tp[region_idx] = torch.zeros_like(
                        input_tp[region_idx], device=self.device
                    )
            # Init states are always zeros
            init_tp[region_idx] = torch.zeros_like(sample_data[0], device=self.device)
        init_tp = TensorPack([init_tp])

        return init_tp, input_tp

    def parse_last_state(self, state_tp: TensorPack):
        """
        Parse the last state of the trajectory
        """
        init_tp = nn4n.empty_tp(len(self.region_dict))
        for region_idx in range(len(self.region_dict)):
            init_tp[region_idx] = state_tp[region_idx][:, -1].detach().clone()

        # return TensorPack([init_tp])

    # ------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------
    @staticmethod
    def print_tp(input: TensorPack, init: TensorPack):
        for i in range(input.shape[0]):
            init_data, input_data = init[0][i], input[i]
            print_str = f"Region {i}: "
            print_str += f"Input: {input_data.shape}, " if input_data is not None else "Input: None, "
            print_str += f"Init: {init_data.shape}, " if init_data is not None else "Init: None, "
            print(print_str)

    def forward(self, inputs: TensorPack, init_states: Optional[TensorPack] = None):
        # Process the inputs and init states
        inputs = torch.cat(inputs.to_list(), dim=2)
        if init_states is not None:
            _is = init_states[0].to_list()
            init_states = [torch.cat(_is, dim=1)]

        # Forward pass
        state = self.model(inputs, init_states)[1][0]  # outputs, [states]

        # Process the output
        state_tp = nn4n.empty_tp(len(self.region_dict))
        for region_key in self.region_dict.keys():
            region_idx = self.region_dict[region_key]
            state_tp[region_idx] = state[..., self.region_indices(region_key)]
        
        # None is the placeholder for the model output, not used here
        return None, state_tp

    def compute_loss(self, state_tp: TensorPack, label_tp: TensorPack, warmup_ts: int = 0):
        total_loss = 0
        loss_dict, log_dict = {}, {}
        for region_key, region_idx in self.region_dict.items():
            region_loss, region_loss_dict, region_log_dict = self.get_region(region_key).compute_loss(
                state=state_tp[region_idx], 
                label=label_tp[region_idx],
                warmup_ts=warmup_ts
            )
            total_loss += region_loss
            log_dict[region_key] = region_log_dict
            loss_dict[region_key] = region_loss_dict
        return total_loss, loss_dict, log_dict

    # ------------------------------------------------------------
    # model Construction
    # ------------------------------------------------------------
    def _initialize_model(self):
        block_recurrent_layer = BlockRecurrentLayer()
        for region_idx, (region_key, region_cfg) in enumerate(self.hcmec_cfg.modules.items()):
            _block = HCMECBlock(
                region_cfg=region_cfg, 
                rep_seed=self.rep_seed, 
                room_idx=self.room_idx
            )
            block_recurrent_layer.add_recurrent(_block)
            self.region_dict[region_key] = region_idx

        rnn_utils.set_between_block_projections(block_recurrent_layer)

        self.model = RNN(recurrent_layers=[block_recurrent_layer])
        self.model.recurrent_layers[0].assemble(
            spectral_norm=False, 
            self_connection=False, 
            postact_noise=self.postact_noise
        )
        self.model.recurrent_layers[0].assembled_recurrent_layer.projection_layer.requires_grad_(False)  # Input will be direct copy, no grad

    def _compute_pads(self):
        """
        Compute the signal padding for each region. Padding is used to account 
        for the time delay in different regions.
        """
        offsets = [self.model.recurrent_layers[0].get_recurrent(i)._offset for i in range(self.n_blocks)]
        min_offset = min(offsets)
        max_offset = max(offsets)
        for region_key in self.region_dict.keys():
            region = self.get_region(region_key)
            region.compute_pads(min_offset, max_offset)
        self.default_pads = (max_offset, min_offset)

    def get_region(self, region_key):
        return self.model.recurrent_layers[0].get_recurrent(self.region_dict[region_key])
    
    def aggregate_sensory_profiles(self):
        sensory_profiles = {}
        for region_key in self.region_dict.keys():
            region = self.get_region(region_key)
            sensory_profiles.update(region.sensory_profile)
        return sensory_profiles

    # ------------------------------------------------------------
    # State Dict Saving and Loading
    # ------------------------------------------------------------
    def state_dict(self):
        return {
            "type": "rnn_hcmec",
            "model_state_dict": self.model.state_dict(),
            "region_dict": self.region_dict,
            "hcmec_cfg": self.hcmec_cfg,
            "gym_cfg": self.gym_cfg,
            "rep_seed": self.rep_seed,
            "room_idx": self.room_idx,
            "postact_noise": self.postact_noise,
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        singleton = cls(
            hcmec_cfg=state_dict["hcmec_cfg"], 
            rep_seed=state_dict.get("rep_seed", 0), 
            room_idx=state_dict.get("room_idx", 0),
            postact_noise=state_dict.get("postact_noise", 0.0)
        )
        singleton.model.load_state_dict(state_dict["model_state_dict"])
        singleton.region_dict = state_dict["region_dict"]
        singleton.set_gym(state_dict["gym_cfg"])
        return singleton

    def overwrite_config(self, config):
        raise NotImplementedError("Not implemented")
