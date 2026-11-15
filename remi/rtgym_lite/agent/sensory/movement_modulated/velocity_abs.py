import numpy as np
from typing import Union, Any
import matplotlib.pyplot as plt
from ....dataclass import AgentState, Trajectory
from .displacement_abs import DisplacementAbs


class VelocityAbs(DisplacementAbs):
    """
    Absolute velocity signal
    """
    sens_type = 'velocity_abs'
    def __init__(self, arena, **kwargs):
        """
        @kwarg n_cells: number of spatially modulated non-grid/place cellcells
        @kwarg sigma_s: sigma_s for the Gaussian smoothing, default is 0.0 sec
        @kwarg ssigma_s: sigma_s for the spatial smoothing, default is 0.0 sec
        @kwarg normalize: normalize the cell responses
        """
        super().__init__(arena, **kwargs)

        # parameters
        self.sigma_ts = self._t_to_ts(kwargs.get('sigma_s', 0.0))
        self.ssigma_ts = self._t_to_ts(kwargs.get('ssigma_s', 0.0))

        # check parameters
        self._check_params()

    def _check_params(self):
        """ Check parameters """
        assert self.n_cells > 0, "n_cells <= 0"

    @staticmethod
    def _displacement_to_velocity(displacement):
        """
        Convert displacement to velocity
        """
        # displacement: (n_batch, n_time, 2), last dimension is x, y
        # velocity: (n_batch, n_time, 1), the magnitude of the velocity
        velocity = np.linalg.norm(displacement, axis=2)
        velocity = velocity[:, :, np.newaxis]
        return velocity

    def _duplicate_res(self, res):
        """
        Duplicate the velocity to the number of cells
        """
        res = np.tile(res, (1, 1, self.n_cells))
        return res

    def get_response(self, agent_data: Union[AgentState, Trajectory, Any]):
        if isinstance(agent_data, Trajectory):
            vel = self._displacement_to_velocity(agent_data.disp) / self.t_res
            dup_res = self._duplicate_res(vel)
            return self._smooth_res(dup_res)
        elif isinstance(agent_data, AgentState):
            return self._displacement_to_velocity(agent_data.disp) / self.t_res
        else:
            raise ValueError(f"Invalid agent_data type: {type(agent_data)}, must be rtgym.dataclass.Trajectory or rtgym.dataclass.AgentState")

    def get_specs(self):
        specs = super().get_specs()
        return specs

    def vis(self, N=5, max_vel=1.0, n_bins=10, cmap='jet', *args, **kwargs):
        """
        Visualize the spatially modulated cells
        """
        vels = np.linspace(0, max_vel, n_bins).reshape(1, -1, 1)
        dups_res = self._duplicate_res(vels)
        smooth_res = self._smooth_res(dups_res)
        
        fig, axs = plt.subplots(1, N, figsize=(4*N, 4))
        for i in range(N):
            # plot, x axis is the vel, y axis is the response
            v = vels[0, :, 0]
            r = smooth_res[0, :, i]
            axs[i].plot(v, r)
        plt.show()
