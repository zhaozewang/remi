import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Union
import remi.utils as utils
from remi.model import HCMEC

from remi.nn4n.nn import (
    BlockRecurrentLayer, 
    BlockRNN,
    TensorPack,
    LinearLayer,
    RecurrentLayer,
    LeakyLinearLayer
)


class Planner(nn.Module):
    def __init__(
            self, 
            planner_cfg: Dict[str, Any], 
            hcmec_path: Path,
            device: str = "cpu", **kwargs
        ):
        super().__init__()
        self.planner_cfg = planner_cfg
        self.device = torch.device(device)

        self.load_and_construct(hcmec_path)

    # ------------------------------------------------------------
    # Common Utils
    # ------------------------------------------------------------
    def to(self, device: str):
        self.device = torch.device(device)
        self.model.to(self.device)
        self.hcmec.to(self.device)
        return self

    @property
    def gym(self):
        return self.hcmec.gym
    
    @property
    def block_rec(self):
        assert hasattr(self, "model"), "Model has not been constructed yet"
        return self.model.recurrent_layers[0]

    def region_indices(self, region_key):
        # TODO: This may need to be updated
        if region_key == "planner":
            return self.block_rec.block_indices(1)
        elif region_key in self.hcmec.region_dict.keys():
            return self.hcmec.region_indices(region_key)
        else:
            raise ValueError(f"Region key {region_key} is not valid")

    def region_idx(self, region_key):
        if region_key == "planner":
            return self.hcmec.n_blocks + 1
        elif region_key in self.hcmec.region_dict.keys():
            return self.hcmec.region_dict[region_key]
        else:
            raise ValueError(f"Region key {region_key} is not valid")

    def region_dim(self, region_key):
        return self.get_region(region_key).hidden_size

    def region_name(self, region_idx):
        n_total_blocks = self.hcmec.n_blocks + 1
        if region_idx == n_total_blocks:
            return "planner"
        elif region_idx < n_total_blocks:
            return self.hcmec.region_name(region_idx)
        else:
            raise ValueError(f"Region index {region_idx} is out of range")

    def get_region(self, region_key):
        assert hasattr(self, "hcmec"), "HCMEC has not been set yet"
        if region_key == "planner":
            return self.block_rec.get_recurrent(1)
        else:
            hcmec_idx = self.hcmec.region_dict[region_key]
            hcmec_layer = self.hcmec.model.recurrent_layers[0]
            return hcmec_layer.get_recurrent(hcmec_idx)
    
    def switch_gym(self, gym_cfg: Dict[str, Any]):
        """
        Move to a new gym

        **Args:**
        - gym_cfg: The gym configuration
        """
        self.gym.init_arena_map(
            shape=gym_cfg["arena_cfg"]["arena_shape"], 
            **gym_cfg["arena_cfg"]["arena_params"]
        )

    # ------------------------------------------------------------
    # Data Generation and Decoding
    # ------------------------------------------------------------
    def generate_data(
            self, duration: int, batch_size: int, pause_prob: float = 0.2,
            init_gc_noise: float = 0.0
        ):
        # Generate the initial state
        init_pos = self.gym.arena.generate_random_pos(batch_size)
        traj = self.gym.agent.random_traverse(
            duration=duration, 
            batch_size=batch_size, 
            init_pos=init_pos,
            pause_prob=pause_prob
        )
        start_state, goal_state = traj[0], traj[-1]

        # Obtained the responses at the start and goal states
        init_res = torch.zeros((batch_size, self.hcmec.size), device=self.device)
        for key in self.planner_cfg["init_keys"]:
            _init = self.get_region(key).sample_data(self.gym, start_state)
            if key == "gc" and init_gc_noise > 0:
                _init = _init + torch.randn_like(_init) * init_gc_noise
            init_res[:, self.region_indices(key)] = _init
        init_tp = TensorPack([init_res, None])

        input_res = []
        noise_level = 0
        for key in self.planner_cfg["planner_keys"]:
            res = self.get_region(key).sample_data(self.gym, goal_state)
            # noise, same shape gaussian noise with std = noise_level
            noise = torch.randn_like(res) * noise_level
            input_res.append(res + noise)

        loss_res = []
        for key in self.planner_cfg["loss_keys"]:
            loss_res.append(self.get_region(key).sample_data(self.gym, goal_state))

        # Convert to training data
        Ts = int(len(traj))
        input_tp = TensorPack([None, torch.cat(input_res, dim=1).unsqueeze(1).repeat(1, Ts, 1)])
        goal_res = torch.cat(loss_res, dim=1).unsqueeze(1).repeat(1, Ts, 1)
        init_tp = TensorPack([init_tp])

        return init_tp, input_tp, goal_res, traj
    
    def parse_response(self, response: TensorPack, key: str):
        """
        Parse the response from the planner

        **Args:**
        - response: The response from the planner, a TensorPack with two elements
        - key: The key to parse the response
        """
        return response[0][..., self.region_indices(key)]

    # ------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------
    def forward(self, input_tp: TensorPack, init_tp: TensorPack):
        return self.model(input_tp, init_tp)

    def extact_hcmec_states(self, network_state: torch.Tensor):
        hcmec_states = []
        for key in self.planner_cfg["loss_keys"]:
            hcmec_states.append(network_state[..., self.region_indices(key)])
        return torch.cat(hcmec_states, dim=2)

    def _planning_loss(self, hcmec_states: torch.Tensor, goal_res: torch.Tensor):
        """
        Compute the planning loss by generating a distribution of the time steps 
        and then weighting the loss by the distribution. The intuition is that 
        the closer to the goal, the more important the loss is.
        """
        alpha = self.planner_cfg["loss_cfg"]["plan"]["alpha"]
        beta = self.planner_cfg["loss_cfg"]["plan"]["beta"]

        diff = hcmec_states - goal_res

        device = hcmec_states.device
        indices = torch.arange(diff.shape[1], device=device) / diff.shape[1]
        exp_values = indices**alpha + beta
        plan_weight = exp_values / torch.sum(exp_values)
        plan_weight = plan_weight.view(1, -1, 1)

        return (torch.pow(diff, 2) * plan_weight).mean() * diff.shape[1]

    def compute_loss(
            self, 
            state_tp: TensorPack, 
            goal_res: torch.Tensor
        ):
        network_state = state_tp[0]
        # Aggregate the planner region states
        hcmec_states = self.extact_hcmec_states(network_state)
        vel_states = network_state[..., self.region_indices("vel")]

        loss_dict = {}
        mse_loss = torch.nn.MSELoss()
        if "vel" in self.planner_cfg["loss_cfg"] and self.planner_cfg["loss_cfg"]["vel"]["lambda"] > 0:
            _l = self.planner_cfg["loss_cfg"]["vel"]["lambda"]
            loss_dict["vel"] = vel_states.mean(dim=(1, 2)).pow(2).mean() * _l

        if "max_vel" in self.planner_cfg["loss_cfg"] and self.planner_cfg["loss_cfg"]["max_vel"]["lambda"] > 0:
            _l = self.planner_cfg["loss_cfg"]["max_vel"]["lambda"]
            mean_vel = vel_states.mean(dim=(0, 1), keepdim=True) # (1, 1, D)
            norm_vel = vel_states / (mean_vel + 1e-6)
            loss_dict["max_vel"] = (norm_vel - 1.0).pow(2).mean() * _l

        if "vel_std" in self.planner_cfg["loss_cfg"] and self.planner_cfg["loss_cfg"]["vel_std"]["lambda"] > 0:
            _l = self.planner_cfg["loss_cfg"]["vel_std"]["lambda"]
            var_per_trial_neuron = vel_states.var(dim=1)  # shape: (B, D)
            std_across_neurons = var_per_trial_neuron.std(dim=1)  # shape: (B,)
            loss_dict["vel_std"] = std_across_neurons.mean() * _l

        if "enc" in self.planner_cfg["loss_cfg"] and self.planner_cfg["loss_cfg"]["enc"]["lambda"] > 0:
            _l = self.planner_cfg["loss_cfg"]["enc"]["lambda"]
            loss_dict["enc"] = hcmec_states.pow(2).max(dim=1).values.mean() * _l

        if "plan" in self.planner_cfg["loss_cfg"] and self.planner_cfg["loss_cfg"]["plan"]["lambda"] > 0:
            _l = self.planner_cfg["loss_cfg"]["plan"]["lambda"]
            if self.planner_cfg["loss_cfg"]["plan"]["alpha"] > 0:
                loss_dict["plan"] = self._planning_loss(hcmec_states, goal_res) / torch.mean(goal_res) * _l
            else:
                loss_dict["plan"] = mse_loss(hcmec_states, goal_res) / torch.mean(goal_res) * _l

        if "eig" in self.planner_cfg["loss_cfg"] and self.planner_cfg["loss_cfg"]["eig"]["lambda"] > 0:
            weight_mat = self.block_rec.get_recurrent(1).leaky_layer.linear_layer.weight
            v = torch.randn(weight_mat.shape[1], device=weight_mat.device)
            v = v / torch.norm(v)

            for _ in range(5):
                v = weight_mat @ v
                v = v / torch.norm(v)

            # After power iterations, compute the largest eigenvalue
            max_eig_val = torch.dot(v, weight_mat @ v)
            loss_dict["eig"] = (max_eig_val - 1.0).abs()**2 * self.planner_cfg["loss_cfg"]["eig"]["lambda"]

        if "norm" in self.planner_cfg["loss_cfg"] and self.planner_cfg["loss_cfg"]["norm"]["lambda"] > 0:
            _l = self.planner_cfg["loss_cfg"]["norm"]["lambda"]
            loss_dict["norm"] = hcmec_states.pow(2).mean() * _l

        if "goal" in self.planner_cfg["loss_cfg"] and self.planner_cfg["loss_cfg"]["goal"]["lambda"] > 0:
            _l = self.planner_cfg["loss_cfg"]["goal"]["lambda"]
            loss_dict["goal"] = mse_loss(hcmec_states[:, -1], goal_res[:, -1]) / torch.mean(goal_res) * _l

        return sum(loss_dict.values()), loss_dict

    # ------------------------------------------------------------
    # Model Construction
    # ------------------------------------------------------------
    def load_and_construct(self, hcmec_path: Path):
        """
        Load the hcmec and construct the planner, if the hcmec is already loaded,
        it will be switched to the new hcmec if the name and version are different

        **Args:**
        - hcmec_path: The path to the hcmec ckpt (the path to .pth file)
        """
        # Load the hcmec
        hcmec_dict = torch.load(hcmec_path, map_location=self.device, weights_only=False)["model"]

        if hcmec_dict["type"] == "rnn_hcmec":
            hcmec = HCMEC.from_state_dict(hcmec_dict)
        else:
            raise ValueError(f"HCMEC type {hcmec_dict['type']} is not valid")
        hcmec.to(self.device)
        hcmec.freeze()
        
        h2p_idx, p2h_idx = self._construct_planner(hcmec)
        
        utils.printc(f"Setting HCMEC to {hcmec_path}", 'g')
        self.hcmec_path = hcmec_path
        self.hcmec = hcmec
        self.h2p_idx = h2p_idx
        self.p2h_idx = p2h_idx

        self.hcmec.gym.agent.behavior.avoid_boundary_dist = -1


    def _construct_planner(self, hcmec: HCMEC):
        """
        Construct the planner, this will load the pre-trained hcmec and 
        construct the planner region

        **Args:**
        - hcmec: The pre-trained hcmec to be controlled by the planner
        """
        # Initialize an empty block recurrent layer
        self.n_blocks = 2
        block_recurrent = BlockRecurrentLayer(n_blocks=self.n_blocks)

        # Construct the planner projections
        planner_dim = self.planner_cfg["hidden_dim"]
        lin_h2p, h2p_idx = self._construct_hcmec2planner(hcmec, planner_dim)
        lin_p2h, p2h_idx = self._construct_planner2hcmec(hcmec, planner_dim)
        planner_block = self._construct_planner_block(hcmec, planner_dim)
        hcmec_block = hcmec.model.recurrent_layers[0].assembled_recurrent_layer

        # Set the block recurrent layer
        hcmec_block.projection_layer = None
        block_recurrent.set_recurrent(idx=0, recurrent_layer=hcmec_block)
        block_recurrent.set_recurrent(idx=1, recurrent_layer=planner_block)
        block_recurrent.set_projection(from_idx=1, to_idx=0, projection_layer=lin_p2h)
        block_recurrent.set_projection(from_idx=0, to_idx=1, projection_layer=lin_h2p)

        # Set the model
        self.model = BlockRNN(recurrent_layers=[block_recurrent])
        self.model.to(self.device)
        return h2p_idx, p2h_idx

    def _reconstruct_planner(self, hcmec: HCMEC):
        """
        Reconstruct the planner, this will load the pre-trained hcmec and 
        reconstruct the planner region

        **Args:**
        - hcmec: The pre-trained hcmec to be controlled by the planner
        """
        # Move the new hcmec to the current block recurrent layer
        assembled_hcmec = hcmec.model.recurrent_layers[0].assembled_recurrent_layer
        assembled_hcmec.projection_layer = None
        self.block_rec.set_recurrent(
            idx=0, 
            recurrent_layer=assembled_hcmec
        )

        # Reconstruct the projections
        planner_dim = self.planner_cfg["hidden_dim"]
        n_h2p, h2p_idx = self._construct_hcmec2planner(hcmec, planner_dim)
        n_p2h, p2h_idx = self._construct_planner2hcmec(hcmec, planner_dim)
        n_h2p.to(self.device)
        n_p2h.to(self.device)

        c_h2p = self.block_rec.get_projection(0, 1)
        c_p2h = self.block_rec.get_projection(1, 0)

        n_h2p.weight.data[:, h2p_idx] = c_h2p.weight.data.clone()[:, self.h2p_idx]
        n_h2p.bias.data = c_h2p.bias.data.clone()  # Planner dim not changed
        n_p2h.weight.data[p2h_idx, :] = c_p2h.weight.data.clone()[self.p2h_idx, :]
        n_p2h.bias.data[p2h_idx] = c_p2h.bias.data.clone()[self.p2h_idx]

        self.block_rec.set_projection(from_idx=0, to_idx=1, projection_layer=n_h2p)
        self.block_rec.set_projection(from_idx=1, to_idx=0, projection_layer=n_p2h)

        return h2p_idx, p2h_idx

    def _construct_planner_block(self, hcmec: HCMEC, planner_dim: int):
        """
        Construct the planner block

        **Args:**
        - hcmec: The hcmec to be controlled by the planner
        - planner_dim: The dimension of the planner
        """
        input_dims = [hcmec.region_dim(key) for key in self.planner_cfg["planner_keys"]]
        rec_lin = LinearLayer(input_dim=planner_dim, output_dim=planner_dim)
        return RecurrentLayer(
            projection_layer=LinearLayer(
                input_dim=sum(input_dims), 
                output_dim=planner_dim
            ),
            leaky_layer=LeakyLinearLayer(
                linear_layer=torch.nn.utils.spectral_norm(rec_lin),
                activation=torch.nn.ReLU(),
                alpha=self.planner_cfg["alpha"],
                learn_alpha=self.planner_cfg["learn_alpha"]
            )
        )

    def _construct_planner2hcmec(self, hcmec: HCMEC, planner_dim: int):
        """
        Planner should control the hcmec by projecting the planner states to the hcmec states
        This is to determine which regions are being controlled by (projected from) the planner

        **Args:**
        - hcmec: The hcmec to be controlled by the planner
        - planner_dim: The dimension of the planner
        """
        # Construct the sparsity mask
        sparsity_mask = torch.zeros(hcmec.size, planner_dim)
        ckeys = self.planner_cfg["control_keys"]
        p2h_idx = torch.cat([hcmec.region_indices(k) for k in ckeys])
        sparsity_mask[p2h_idx, :] = 1  # Set these indices to 1

        # Construct the linear layer
        return LinearLayer(
            input_dim=planner_dim,
            output_dim=hcmec.size,
            sparsity_mask=sparsity_mask
        ), p2h_idx

    def _construct_hcmec2planner(self, hcmec: HCMEC, planner_dim: int):
        """
        Planner should control the hcmec by projecting the planner states to the hcmec states
        This is to determine which regions project to the planner

        **Args:**
        - hcmec: The hcmec to be controlled by the planner
        - planner_dim: The dimension of the planner
        """
        # Construct the sparsity mask
        sparsity_mask = torch.zeros(planner_dim, hcmec.size)
        pkeys = self.planner_cfg["planner_keys"]
        h2p_idx = torch.cat([hcmec.region_indices(k) for k in pkeys])
        sparsity_mask[:, h2p_idx] = 1  # Set these indices to 1

        # Construct the linear layer
        return LinearLayer(
            input_dim=hcmec.size,
            output_dim=planner_dim,
            sparsity_mask=sparsity_mask
        ), h2p_idx

    # ------------------------------------------------------------
    # Model Saving and Loading
    # ------------------------------------------------------------
    def state_dict(self):
        return {
            "type": "rnn_planner",
            "model_state_dict": self.model.state_dict(),
            "planner_cfg": self.planner_cfg,
            "hcmec_path": self.hcmec_path,
            "h2p_idx": self.h2p_idx,
            "p2h_idx": self.p2h_idx
        }

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Any], device: str = "cpu"):
        planner = cls(
            planner_cfg=state_dict["planner_cfg"], 
            hcmec_path=state_dict["hcmec_path"],
            device=device
        )
        planner.h2p_idx = state_dict["h2p_idx"]
        planner.p2h_idx = state_dict["p2h_idx"]
        planner.model.load_state_dict(state_dict["model_state_dict"])
        planner.model.to(device)
        return planner
