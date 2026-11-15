import numpy as np
import hashlib
import pickle
from numpy.random import default_rng
from typing import Union
from ....utils import print_dict, visualize_fields
from ....dataclass import AgentState, Trajectory

class SMBase():
    """ Spatially modulated behavior base class """
    sens_category = 'spatial_modulated'
    sens_type = 'sm_base'
    def __init__(self, arena, n_cells, sensory_key, seed=None, **kwargs):
        """
        @kwarg arena: arena object
        @kwarg seed: seed for random number generator
        """
        self.arena = arena
        self.n_cells = n_cells
        self.sensory_key = sensory_key

        # Initialize random number generator
        if seed is not None:
            seed = self._hash_seed(seed, sensory_key)
        self.rng = default_rng(seed)

        # check if the class is the base class
        assert type(self) != SMBase, "SMBase is an abstract class"
        assert self.n_cells > 0, "n_cells <= 0"

    @staticmethod
    def _hash_seed(seed, sensory_key):
        return seed + int(hashlib.sha256(sensory_key.encode('utf-8')).hexdigest(), 16) % (10 ** 8)

    def _init_response_map(self):
        # border padding is also included in the response field
        self.response_map = np.zeros((self.n_cells, *self.arena.dimensions))

    def get_specs(self):
        return {
            'n_cells': self.n_cells,
            'response_field_width (with 5 pixels padding)': self.response_map.shape[1],
            'response_field_height (with 5 pixels padding)': self.response_map.shape[2],
        }

    def print_specs(self):
        """ Print specs """
        print_dict(self.get_specs())

    def get_response(self, agent_data: Union[AgentState, Trajectory]):
        """
        Get response_map

        Parameters:
            - traj: rtgym.dataclass.Trajectory object

        Returns:
            - response_map: spatially modulated responses. The response_map are of shape (n_cells, *arena_dimensions).
                            After indexing, it will be of shape (n_cells, n_batch, n_timesteps).
                            When return, reshape it to (n_batch, n_timesteps, n_cells).
        """
        if isinstance(agent_data, Trajectory):
            return self.response_map[:, agent_data.int_coord[..., 0], agent_data.int_coord[..., 1]].transpose(1, 2, 0)
        elif isinstance(agent_data, AgentState):
            return self.response_map[:, agent_data.int_coord[:, 0], agent_data.int_coord[:, 1]].transpose(1, 0)
        else:
            raise ValueError(f"Invalid agent_data type: {type(agent_data)}, must be rtgym.dataclass.Trajectory or rtgym.dataclass.AgentState")

    def vis(self, N=10, cmap='jet', *args, **kwargs):
        """
        Visualize the spatially modulated cells
        """
        cells = self.response_map[:N]
        return visualize_fields(cells, cmap=cmap, mask=self.arena.inv_arena_map)

    def save(self, file_path):
        """
        Save the object to a file, excluding dynamically generated data.
        
        Parameters:
            - file_path: Path to the file where the object will be saved.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self.state_dict(), f)


    @classmethod
    def load(cls, file_path, arena):
        """
        Load the GridCell object from a file and reconstruct it.

        Parameters:
        - file_path (str): Path to the saved file.
        - arena (Arena): Arena object to reinitialize the class.

        Returns:
        - Reconstructed GridCell object.
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        return cls.load_from_dict(data, arena)
