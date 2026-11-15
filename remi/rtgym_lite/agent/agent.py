import numpy as np
from .behavior import Behavior
from .sensory import Sensory
from ..dataclass import AgentState, Trajectory, RawAgentState, RawTrajectory
from typing import Union


class Agent():
    def __init__(self, gym):
        self.gym = gym
        self.behavior_profile = None
        self.sensory_profile = None
        self.sensory = Sensory(self.gym)
        self.behavior = Behavior(self.gym)
        self._state = RawAgentState()

    @property
    def temporal_resolution(self):
        return self.gym.temporal_resolution

    @property
    def spatial_resolution(self):
        return self.gym.spatial_resolution

    @property
    def sensories(self):
        return self.sensory.sensories

    @property
    def arena(self):
        return self.gym.arena
    
    @property
    def controllable(self):
        return self.behavior.controllable

    @property
    def autonomous(self):
        return self.behavior.autonomous

    @property
    def state(self):
        return self._state

    def _on_arena_change(self):
        if self.arena is not None:
            self.behavior._on_arena_change()
            self.sensory._on_arena_change()
            self._init_behavior_from_profile()
            self._init_sensory_from_profile()

    def set_behavior(self, behavior_profile):
        self.behavior_profile = behavior_profile
        self._init_behavior_from_profile()

    def _init_behavior_from_profile(self):
        """ 
        Initialize behavior from behavior profile
        """
        if self.behavior_profile is not None and self.arena is not None:
            self.behavior.init_from_profile(self.behavior_profile)

    def set_sensory(self, sensory_profile):
        self.sensory_profile = sensory_profile
        self._init_sensory_from_profile()

    def add_sensory(self, sensory_profile):
        self.sensory_profile.update(sensory_profile)
        self.sensory.add_sensory(sensory_profile)

    def set_sensory_manually(self, sens_type, sens):
        assert False, "TODO: some error to be fixd"
        if self.sensory is not None:
            print('Warning: sensory is reset')
        self.sensory.set_sensory_manually(sens_type, sens)
        self.sensory = sensory

    @property
    def sensories(self):
        return self.sensory.sensories

    def _init_sensory_from_profile(self):
        """
        Initialize sensory from sensory profile
        """
        if self.sensory_profile is not None and self.arena is not None:
            self.sensory.init_from_profile(self.sensory_profile)

    # ================================
    # Behavior
    # ================================
    def random_traverse(self, duration: float, batch_size: int, init_pos=None, init_state=None, pause_prob=0):
        traj, state = self.behavior.generate_trajectory(duration, batch_size, init_pos, init_state)
        if pause_prob > 0:
            pause_mask = np.random.rand(batch_size) < pause_prob
            traj.disp[pause_mask] = np.zeros_like(traj.disp[pause_mask])
            traj.coord[pause_mask] = traj.coord[pause_mask, 0][:, np.newaxis]
            traj.hd[pause_mask] = np.zeros_like(traj.hd[pause_mask])
            # print("Continuing trajectory maybe deprecated when pause_prob > 0")
        self._state = state
        return traj

    def step(self, displacement):
        self._state = self.controllable.step(self._state, displacement)

    def get_response(
            self,
            agent_data: Union[AgentState, Trajectory],
            return_format='array', 
            keys=None, 
            str_filter=None, 
            type_filter=None
        ):
        if isinstance(agent_data, RawAgentState):
            agent_data = agent_data.to_agent_state()
        elif isinstance(agent_data, RawTrajectory):
            agent_data = agent_data.to_trajectory()
        return self.sensory.get_response(agent_data, return_format, keys, str_filter, type_filter)

    def spawn(self, init_pos=None, init_state=None):
        """
        Spawn the controller at the given position and state.

        **Parameters**:
            - init_pos: initial position of the controller
            - init_state: initial state of the controller, if both init_pos and init_state are provided, 
                    init_state will be used. The init_state could be either a rtgym.dataclass.AgentState object
                    or a rtgym.dataclass.RawAgentState object.

        """
        self.controllable.reset()
        if init_state is not None:
            if isinstance(init_state, RawAgentState):
                self._state = init_state.copy()
            elif isinstance(init_state, AgentState):
                self._state.reset()
                self._state.coord = init_state.coord
                self._state.disp = init_state.disp
        elif init_pos is not None:
            self._state.reset()
            self._state.coord = init_pos

    def random_spawn(self, batch_size: int):
        """
        Spawn the controller at a random position in the arena.
        """
        self.spawn(init_pos=self.arena.random_position(batch_size))

    def step(self, displacement):
        self._state = self.controllable.step(self._state, displacement)
