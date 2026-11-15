"""
RatatouGym is a python package that provides a simple interface to generate sensory responses
from a virtual agent in a virtual environment.

This file contains more documentation as it serves as the main API for the package.
"""

import os
import numpy as np
from .agent import Agent
from .arena import Arena
import torch
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as animation


class RatatouGym():
    def __init__(
            self,
            temporal_resolution,
            spatial_resolution,
            **kwargs
        ):
        self.temporal_resolution = temporal_resolution
        self.spatial_resolution = spatial_resolution
        self.arena = Arena(self)
        self.agent = Agent(self)

        self.arena.subscribe(self._on_arena_change)

    # Basic get methods without trial generation
    # ======================================================================================
    @property
    def t_res(self):
        return self.temporal_resolution

    @property
    def s_res(self):
        return self.spatial_resolution

    def to_ts(self, t):
        if isinstance(t, (int, float)):
            return int(np.round(t * 1e3 / self.t_res))
        elif isinstance(t, np.ndarray):
            return np.round(t * 1e3 / self.t_res).astype(int)
        elif 'torch' in str(type(t)):
            return torch.round(t * 1e3 / self.t_res).int()
        else:
            raise TypeError("Unsupported type for t")

    def to_sec(self, ts):
        if isinstance(ts, (int, float)):
            return ts * self.t_res / 1e3
        elif isinstance(ts, np.ndarray):
            return ts * self.t_res / 1e3
        elif 'torch' in str(type(ts)):
            return ts * self.t_res / 1e3
        else:
            raise TypeError("Unsupported type for ts")

    def to_coord(self, pos):
        return np.array(pos/self.s_res)

    @property
    def arena_map(self):
        return self.arena.arena_map
    
    @property
    def inv_arena_map(self):
        return self.arena.inv_arena_map

    def random_pos(self, n):
        return self.arena.generate_random_pos(n)
    # ======================================================================================
    
    # Set methods to initialize the arena and agent
    # ======================================================================================
    def _on_arena_change(self):
        self.agent._on_arena_change()

    def load_arena(self, path):
        """
        Load an arena from a file
        """
        self.arena.load(path)

    def save_arena(self, path):
        """
        Save the arena to a file
        """
        # file extension is npz
        assert os.path.splitext(path)[1] == '.npz', "file extension must be .npz"
        self.arena.save(path)

    def init_arena_map(self, **kwargs):
        """
        Initialize the arena map with a predefined shape
        """
        self.arena.init_arena_map(**kwargs)

    def set_arena_map(self, arena_map):
        """
        Set arena map with custom map (numpy array; 0: free space, 1: wall)

        **Parameters**:
            - arena_map: numpy array of shape (H, W)
        """
        self.arena.set_arena_map(arena_map=arena_map)
    # ======================================================================================
    
    # Sensory methods
    # ======================================================================================
    def set_sensory_manually(self, sens_type, sensory):
        assert False, "TODO: temporary disabled"
        self.agent.set_sensory_manually(sens_type, sensory)
    # ======================================================================================

    # # Trial generation and methods
    # # ======================================================================================
    # def new_trial(self, *args, **kwargs):
    #     """
    #     This will only generate a trajectory, the trajectory can always be reused 
    #     if the arena (not sensory) remain to be the same and thus saves generation time.

    #     **Parameters**:
    #         - duration: duration of the trial in seconds
    #         - batch_size: number of trials to generate
    #         - init_pos: initial position of the agent
    #     """
    #     return self.trial.new_trial(*args, **kwargs)
    
    # def continue_trial(self, *args, **kwargs):
    #     """
    #     Continue the trial with the same agent and arena.
    #     """
    #     return self.trial.continue_trial(*args, **kwargs)

    # def save_trial(self, path, file_name):
    #     """
    #     Save the generated trial to a file

    #     **Parameters**:
    #         - path: path to save the trial
    #     """
    #     self.trial.save_trial(path, file_name)

    # def load_trial(self, path):
    #     """
    #     Load a trial from a file

    #     Parameters:
    #         - path: path to load the trial
    #     """
    #     self.trial.load_trial(path)

    # def vis_traj(self, *args, **kwargs):
    #     return self.trial.vis_traj(*args, **kwargs)

    # def vis_sensory(self, *args, **kwargs):
    #     return self.trial.vis_sensory(*args, **kwargs)

    @staticmethod
    def _compute_plot_dimensions(arena_map, fig_height):
        aspect_ratio = arena_map.shape[1] / arena_map.shape[0]
        fig_width = fig_height * aspect_ratio
        return fig_width, fig_height

    def vis_gif(self, traj, plot_w, plot_h, return_format='anim', stride=10, interval=50, max_frames=100):
        """
        Create an animated visualization of the trajectory.
        
        Parameters:
            - traj: trajectory data
            - plot_w, plot_h: width and height of the plot
            - return_format: 'anim' or 'html' for animation format
            - stride: step size for sampling trajectory points (higher = faster)
            - interval: milliseconds between frames (lower = faster playback)
            - max_frames: maximum number of frames to generate (lower = faster generation)
        """
        if traj.int_coord.shape[1] == 0 or traj.hd.shape[1] == 0:
            raise ValueError("traj.int_coord or traj.hd has no data along the expected axis.")

        plot_w, plot_h = self._compute_plot_dimensions(self.arena.arena_map, plot_h)
        fig, ax = plt.subplots(figsize=(plot_w, plot_h))
        ax.imshow(self.inv_arena_map)
        ax.axis('off')
        line, = ax.plot([], [], 'r')
        
        # Initialize the quiver arrow
        arrow = ax.quiver(0, 0, 0.05, 0.05, angles='xy', scale_units='xy', scale=0.2, color='blue')

        # Precompute trajectory data
        traj_length = traj.int_coord.shape[1]
        
        # Calculate appropriate stride to ensure we cover the full trajectory
        total_frames = min(max_frames, int(np.ceil(traj_length / stride)))
        frames_stride = max(1, int(np.ceil(traj_length / total_frames)))
        
        # Precompute frame indices to ensure we show the full trajectory
        frame_indices = np.linspace(0, traj_length-1, total_frames, dtype=int)
        
        def init():
            line.set_data([], [])
            arrow.set_offsets([0, 0])
            arrow.set_UVC(0.05, 0.05)
            return line, arrow

        def animate(i):
            frame_idx = frame_indices[i]
            
            # Always show the complete trajectory up to the current point
            line.set_data(traj.float_coord[0, :frame_idx+1, 1], traj.float_coord[0, :frame_idx+1, 0])

            if frame_idx > 0:
                # Get the tip position of the line
                tip_x, tip_y = traj.float_coord[0, frame_idx, 1], traj.float_coord[0, frame_idx, 0]
                
                # Calculate direction based on the angle
                angle = traj.hd[0, frame_idx, 0]
                direction_x = np.cos(angle)
                direction_y = np.sin(angle)
                
                # Update arrow position and direction
                arrow.set_offsets([tip_x, tip_y])
                arrow.set_UVC(direction_x, direction_y)
            
            return line, arrow

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=total_frames, interval=interval, blit=True)
        plt.tight_layout()

        return HTML(anim.to_jshtml()) if return_format == 'html' else anim

    def vis_traj(self, traj, cmap="viridis", linewidth=1, return_format=None, height=3, ax=None, vis_kwargs=None):
        """
        Visualize the trajectory of the agent.

        Parameters:
            - traj: rtgym.dataclass.Trajectory object
            - return_format: format of the returned visualization. Can be 'anim', 'html' or None.
                When 'anim'/'html', vis as an animation. Otherwise, vis as a static plot.
            - fig_height: height of the figure in inches.
        """

        plot_w, plot_h = self._compute_plot_dimensions(self.arena.arena_map, height)
        if return_format == 'anim' or return_format == 'html':
            return self.vis_gif(traj, plot_w, plot_h, return_format, **vis_kwargs)
        else:
            fig, ax = plt.subplots(figsize=(plot_w, plot_h))
            ax.imshow(self.arena.inv_arena_map, cmap=cmap)
            ax.plot(traj.float_coord[:, :, 1].T, traj.float_coord[:, :, 0].T, color='red', linewidth=linewidth)
            ax.set_axis_off()
            return fig, ax
