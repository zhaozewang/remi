"""
The sensory module contains the Sensory class which is responsible for 
creating and managing spatially and movement-modulated sensory cells of the agent.
"""


import pickle
import numpy as np
import torch
from typing import Dict, Any, Union, List, Tuple
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
import faiss

from ...dataclass import AgentState, Trajectory
from .spatial_modulated import *
from .movement_modulated import *


class Sensory:
    """
    Sensory class for spatially and movement-modulated sensory cells.
    This class is usually not directly initialized but initialized by the rtgym.

    **Parameters**
        gym: rtgym.rtgym.RatatouGym object
    """
    def __init__(self, gym):
        self.gym = gym
        self.sensories = {}
        self.ranges = None  # Keep track of the indices of the sensory cells

    @property
    def t_res(self):
        return self.gym.t_res

    @property
    def s_res(self):
        return self.gym.s_res
    
    @property
    def arena(self):
        return self.gym.arena
    
    @property
    def common_params(self):
        return {'arena': self.arena, 't_res': self.t_res}

    def _on_arena_change(self):
        """
        Set the arena for the sensory cells.

        Parameters:
            arena: rtgym.arena.Arena object
        """
        pass

    def list_all(self):
        """
        List all the sensory cells.
        """
        return list(self.sensories.keys())

    def init_from_profile(self, sensory_profile):
        # Initialize spatial and movement modulated cells
        self.sensories = {}
        self._update_sensories(sensory_profile if sensory_profile is not None else {})
        self._update_ranges()
    
    def add_sensory(self, sensory_profile: Dict[str, Any]):
        """
        Add a sensory cell to the sensory system.

        Parameters:
            sensory_profile: dictionary containing the sensory profile.
        """
        self._update_sensories(sensory_profile if sensory_profile is not None else {})
        self._update_ranges()

    def _update_sensories(self, profile_list):
        """
        Initializes the cells based on the provided profile list.
        """
        for key, value in profile_list.items():
            sensory_class = Sensory._get_sensory_class(value['type'])
            self.sensories[key] = sensory_class(
                sensory_key=key, 
                **self.common_params, 
                **value
            )
    
    def _update_ranges(self):
        _ranges = np.cumsum([_sens.n_cells for _sens in self.sensories.values()])
        _ranges = np.insert(_ranges, 0, 0).tolist()
        self.ranges = {key: (_ranges[i], _ranges[i+1]) for i, key in enumerate(self.sensories.keys())}
    
    def filter_sensories(self, keys=None, str_filter=None, type_filter=None):
        """
        This helps to find the keys of the sensory cells that match the given criteria.

        It will prioritize the most specific filter. The specificity from most to least is:
            keys > str_filter > type_filter
        """
        if keys is not None:
            if isinstance(keys, str):
                return_keys = [keys]
            elif isinstance(keys, list):
                return_keys = keys
            else:
                raise ValueError(f"Unknown keys: {keys}")
        elif str_filter is not None:
            return_keys = [key for key in self.sensories.keys() if str_filter in key]
        elif type_filter is not None:
            return_keys = [key for key, sensory_item in self.sensories.items() if type_filter == sensory_item.sens_type]
        else:
            return_keys = list(self.sensories.keys())
        return sorted(return_keys)

    def num_sensories(self, keys=None, str_filter=None, type_filter=None):
        keys = self.filter_sensories(keys, str_filter, type_filter)
        return sum([self.sensories[key].n_cells for key in keys])

    @staticmethod
    def _get_sensory_class(sensory_type):
        sensory_classes = {cls.sens_type: cls for cls in 
                           [WeakSMCell, VelocityAbs, DirectionCell, GridCell]}

        assert sensory_type in sensory_classes, f"Unknown sensory type: {sensory_type}"
        return sensory_classes[sensory_type]
    
    def aggregate_res_maps(self, keys=None, str_filter=None, type_filter=None):
        """
        Aggregate the sensory response maps.

        **Parameters**
            keys : list or None, optional
                Keys to filter the sensories.
            str_filter : optional
                String filter for sensories.
            type_filter : optional
                Type filter for sensories.

        **Returns**
            res_maps : np.ndarray
                The aggregated sensory response maps.
        """
        # Filter and aggregate the sensory response maps.
        keys = self.filter_sensories(keys, str_filter, type_filter)
        res_maps = []
        for key in keys:
            assert self.sensories[key].sens_category == 'spatial_modulated', (
                "Only spatial_modulated sensory cells can be decoded into a trajectory"
            )
            res_maps.append(self.sensories[key].response_map)
        
        # Concatenate along the cell dimension. Shape: (n_cells, H, W)
        return np.concatenate(res_maps, axis=0)

    def decode_response(self, response: np.ndarray, res_maps=None, keys=None, str_filter=None, type_filter=None, 
                        use_torch=True, device=None, method="euclidean", **kwargs):
        """
        Decode the sensory response into either a trajectory of 2D coordinates or a single AgentState.

        **Parameters**
            response : np.ndarray
                Array of shape (n_batch, n_timesteps, n_cells) or (n_batch, n_cells)
                representing the sensory response.
            res_maps : np.ndarray, optional
                Precomputed response maps to use for decoding.
            keys : list or None, optional
                Keys to filter the sensories.
            str_filter : optional
                String filter for sensories.
            type_filter : optional
                Type filter for sensories.
            use_torch : bool, optional
                Whether to use PyTorch for faster computation (default: True).
            device : str or torch.device, optional
                Device to use for PyTorch computation. If None, uses CUDA if available, else CPU.
            method : str, optional
                Method to use for finding the closest coordinate. Options are:
                - "euclidean": Use squared Euclidean distance (default)
                - "torch_euclidean": Use PyTorch for Euclidean distance computation
                - "kdtree": Use K-d tree for nearest neighbor search (fastest for high-dimensional data)
                - "lsh": Use Locality-Sensitive Hashing for approximate nearest neighbor search
                - "faiss": Use Facebook AI Similarity Search for very fast nearest neighbor search
                - "interpolation": Use spatial interpolation to predict coordinates directly

        **Returns**
            Union[Trajectory, AgentState]
                If input shape is (B,T,D): returns Trajectory with coordinates (B,T,2)
                If input shape is (B,D): returns AgentState with coordinates (B,2)
        """
        # Select the appropriate method based on parameters
        if method == "kdtree":
            return self._decode_response_kdtree(response, res_maps, keys, str_filter, type_filter)
        elif method == "torch_euclidean" and use_torch and torch.cuda.is_available():
            return self._decode_response_torch(response, res_maps, keys, str_filter, type_filter, device)
        elif method == "faiss":
            return self._decode_response_faiss(response, res_maps, keys, str_filter, type_filter, **kwargs)
        elif method == "interpolation":
            return self._decode_response_interpolation(response, res_maps, keys, str_filter, type_filter, **kwargs)
        
        # Fall back to numpy implementation if torch not used or not available
        if res_maps is None:
            res_maps = self.aggregate_res_maps(keys, str_filter, type_filter)
        n_cells, H, W = res_maps.shape

        # Create a coordinate map and reshape res_maps to shape (H*W, n_cells)
        M = res_maps.reshape(n_cells, -1).T  # shape: (H*W, n_cells)
        # Replace NaNs with infinities so they are not selected in the min distance computation.
        M = np.nan_to_num(M, nan=np.inf)

        coords = np.stack(
            np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), axis=-1
        ).reshape(-1, 2)  # shape: (H*W, 2)

        # Handle both 2D and 3D input shapes for the response
        if len(response.shape) == 3:
            B, T, _ = response.shape
            r_flat = response.reshape(-1, n_cells)  # shape: (B*T, n_cells)
        else:  # response.shape == (B, n_cells)
            B = response.shape[0]
            r_flat = response  # shape: (B, n_cells)

        # Compute squared Euclidean distances.
        dists = np.sum((r_flat[:, None, :] - M[None, :, :]) ** 2, axis=-1)
        min_indices = np.argmin(dists, axis=-1)
        pred_coords = coords[min_indices]

        # Return appropriately typed result
        if len(response.shape) == 3:
            pred_coords = pred_coords.reshape(B, T, 2)
            return Trajectory(coord=pred_coords)
        else:
            return AgentState(coord=pred_coords)
            
    def _decode_response_kdtree(self, response, res_maps=None, keys=None, str_filter=None, type_filter=None):
        """
        K-d tree implementation of decode_response for faster nearest neighbor search.
        K-d trees are especially efficient for high-dimensional data compared to brute force methods.
        """
        # Get response maps if not provided
        if res_maps is None:
            res_maps = self.aggregate_res_maps(keys, str_filter, type_filter)
            
        n_cells, H, W = res_maps.shape
        
        # Reshape response maps to (H*W, n_cells)
        M = res_maps.reshape(n_cells, -1).T  # shape: (H*W, n_cells)
        
        # Replace NaNs with infinities
        M = np.nan_to_num(M, nan=np.inf)
        
        # Create coordinate map
        coords = np.stack(
            np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), axis=-1
        ).reshape(-1, 2)  # shape: (H*W, 2)
        
        # Process based on input shape
        is_trajectory = len(response.shape) == 3
        if is_trajectory:
            B, T, _ = response.shape
            r_flat = response.reshape(-1, n_cells)  # shape: (B*T, n_cells)
        else:
            B = response.shape[0]
            r_flat = response  # shape: (B, n_cells)
        
        # Build K-d tree on feature space - this is the most time-intensive part
        # but it's only built once and then reused for all queries
        kdtree = KDTree(M)
        
        # Find nearest neighbor for each response vector
        # This returns both distances and indices, but we only need indices
        _, min_indices = kdtree.query(r_flat, k=1)
        
        # Flatten the indices array
        min_indices = min_indices.flatten()
        
        # Get the corresponding coordinates
        pred_coords = coords[min_indices]
        
        # Reshape based on input
        if is_trajectory:
            pred_coords = pred_coords.reshape(B, T, 2)
            return Trajectory(coord=pred_coords)
        else:
            return AgentState(coord=pred_coords)
    
    def _decode_response_torch(self, response, res_maps=None, keys=None, str_filter=None, type_filter=None, device=None):
        """
        PyTorch implementation of decode_response for faster computation.
        """
        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
            
        # Get response maps if not provided
        if res_maps is None:
            res_maps = self.aggregate_res_maps(keys, str_filter, type_filter)
            
        # Convert numpy arrays to torch tensors
        if isinstance(res_maps, np.ndarray):
            res_maps = torch.from_numpy(res_maps).to(device)
        if isinstance(response, np.ndarray):
            response = torch.from_numpy(response).to(device)
            
        n_cells, H, W = res_maps.shape
        
        # Create coordinate map
        y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), 
                                          torch.arange(W, device=device), 
                                          indexing='ij')
        coords = torch.stack([y_coords, x_coords], dim=-1).reshape(-1, 2)
        
        # Reshape response maps to (H*W, n_cells)
        M = res_maps.reshape(n_cells, -1).t()
        
        # Replace NaNs with infinities
        M = torch.nan_to_num(M, nan=float('inf'))
        
        # Process based on input shape
        is_trajectory = len(response.shape) == 3
        if is_trajectory:
            B, T, _ = response.shape
            r_flat = response.reshape(-1, n_cells)
        else:
            B = response.shape[0]
            r_flat = response
            
        # Optimized distance computation using torch operations
        # This avoids the memory-intensive broadcasting by computing chunks
        chunk_size = 1024  # Process in chunks to avoid OOM errors
        min_indices = []
        
        for i in range(0, r_flat.size(0), chunk_size):
            chunk = r_flat[i:i+chunk_size]
            # Compute ||r - M||^2 = ||r||^2 - 2<r,M> + ||M||^2
            # We can ignore ||r||^2 since it doesn't affect argmin
            r_norm = (chunk**2).sum(dim=1, keepdim=True)
            m_norm = (M**2).sum(dim=1)
            # Compute the matrix multiplication for <r,M> efficiently
            prod = torch.mm(chunk, M.t())
            # dist = r_norm - 2*prod + m_norm
            dist = r_norm - 2*prod + m_norm
            chunk_min_indices = torch.argmin(dist, dim=1)
            min_indices.append(chunk_min_indices)
            
        min_indices = torch.cat(min_indices)
        pred_coords = coords[min_indices]
        
        # Reshape result based on input shape
        if is_trajectory:
            pred_coords = pred_coords.reshape(B, T, 2)
            # Convert back to numpy for Trajectory class
            return Trajectory(coord=pred_coords.cpu().numpy())
        else:
            # Convert back to numpy for AgentState class
            return AgentState(coord=pred_coords.cpu().numpy())

    def _decode_response_faiss(self, response, res_maps=None, keys=None, str_filter=None, type_filter=None, n_clusters=100):
        """
        Facebook AI Similarity Search (FAISS) implementation for extremely fast nearest neighbor search.
        FAISS is optimized for high-dimensional vectors and can use GPU acceleration.
        
        This implementation uses the IndexIVFFlat index which performs k-means clustering on the data
        and then does exact search only within relevant clusters, dramatically speeding up queries.
        """
        # Get response maps if not provided
        if res_maps is None:
            res_maps = self.aggregate_res_maps(keys, str_filter, type_filter)
            
        n_cells, H, W = res_maps.shape
        
        # Reshape response maps to (H*W, n_cells)
        M = res_maps.reshape(n_cells, -1).T  # shape: (H*W, n_cells)
        
        # Replace NaNs
        M = np.nan_to_num(M, nan=0.0)  # FAISS doesn't handle inf values well
        
        # Ensure data is in float32 format (required by FAISS)
        M = M.astype(np.float32)
        
        # Create coordinate map
        coords = np.stack(
            np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), axis=-1
        ).reshape(-1, 2)  # shape: (H*W, 2)
        
        # Process based on input shape
        is_trajectory = len(response.shape) == 3
        if is_trajectory:
            B, T, _ = response.shape
            r_flat = response.reshape(-1, n_cells)  # shape: (B*T, n_cells)
        else:
            B = response.shape[0]
            r_flat = response
            
        # Ensure query data is in float32
        r_flat = r_flat.astype(np.float32)
        
        # Create the FAISS index
        d = n_cells  # Dimensionality of vectors
        quantizer = faiss.IndexFlatL2(d)  # Base index for exact L2 distance
        
        # Adjust number of clusters based on data size
        n_clusters = min(n_clusters, M.shape[0] // 39)  # FAISS requires at least 39 vectors per cluster
        n_clusters = max(1, n_clusters)  # At least 1 cluster
        
        # Create the IVF index with n_clusters centroids
        # The second parameter (n_lists) is the number of clusters
        index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_L2)
        
        # Train the index on the data
        index.train(M)
        
        # Add vectors to the index
        index.add(M)
        
        # Set the number of clusters to probe at search time
        # Higher values give better accuracy at the cost of speed
        index.nprobe = min(10, n_clusters)
        
        # Search for nearest neighbors
        _, I = index.search(r_flat, 1)  # k=1 for single nearest neighbor
        
        # Get coordinates of nearest neighbors
        min_indices = I.flatten()
        pred_coords = coords[min_indices]
        
        # Reshape based on input
        if is_trajectory:
            pred_coords = pred_coords.reshape(B, T, 2)
            return Trajectory(coord=pred_coords)
        else:
            return AgentState(coord=pred_coords)
    
    def _decode_response_interpolation(self, response, res_maps=None, keys=None, str_filter=None, type_filter=None,
                                      n_anchors=1000, random_state=42):
        """
        Interpolation-based method that directly predicts coordinates from responses.
        
        This approach:
        1. Selects n_anchors representative points from the feature space
        2. Builds a mapping from feature vectors to coordinates for these anchors
        3. For each query, finds k nearest anchors and interpolates their coordinates
        
        This exploits the spatial structure where similar coordinates have similar representations.
        """
        # Get response maps if not provided
        if res_maps is None:
            res_maps = self.aggregate_res_maps(keys, str_filter, type_filter)
            
        n_cells, H, W = res_maps.shape
        
        # Reshape response maps to (H*W, n_cells)
        M = res_maps.reshape(n_cells, -1).T  # shape: (H*W, n_cells)
        
        # Replace NaNs
        M = np.nan_to_num(M, nan=0.0)
        
        # Create coordinate map
        coords = np.stack(
            np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), axis=-1
        ).reshape(-1, 2)  # shape: (H*W, 2)
        
        # Process based on input shape
        is_trajectory = len(response.shape) == 3
        if is_trajectory:
            B, T, _ = response.shape
            r_flat = response.reshape(-1, n_cells)  # shape: (B*T, n_cells)
        else:
            B = response.shape[0]
            r_flat = response
            
        # Limit number of anchors based on data size
        n_anchors = min(n_anchors, M.shape[0])
        
        # Select anchor points using K-means clustering to find representative points
        kmeans = KMeans(n_clusters=n_anchors, random_state=random_state, n_init=10)
        kmeans.fit(M)
        anchor_indices = []
        
        # For each cluster, find the point closest to the centroid
        for i in range(n_anchors):
            cluster_points = M[kmeans.labels_ == i]
            if len(cluster_points) > 0:
                centroid = kmeans.cluster_centers_[i]
                dists = np.sum((cluster_points - centroid) ** 2, axis=1)
                closest_idx = np.argmin(dists)
                # Get original index
                original_indices = np.where(kmeans.labels_ == i)[0]
                anchor_indices.append(original_indices[closest_idx])
            
        # Get anchor features and coordinates
        anchor_features = M[anchor_indices]
        anchor_coords = coords[anchor_indices]
        
        # Build KD-tree for fast nearest neighbor search among anchors
        anchor_tree = KDTree(anchor_features)
        
        # Find k nearest anchors for each query point
        k = min(5, len(anchor_indices))  # Use up to 5 neighbors for interpolation
        dists, indices = anchor_tree.query(r_flat, k=k)
        
        # Compute weights based on distances (inverse distance weighting)
        weights = 1.0 / (dists + 1e-10)  # Add small constant to avoid division by zero
        weights = weights / weights.sum(axis=1, keepdims=True)  # Normalize
        
        # Interpolate coordinates
        pred_coords = np.zeros((r_flat.shape[0], 2))
        for i in range(k):
            pred_coords += weights[:, i:i+1] * anchor_coords[indices[:, i]]
            
        # Round to integer coordinates
        pred_coords = np.round(pred_coords).astype(int)
        
        # Clip to valid ranges
        pred_coords[:, 0] = np.clip(pred_coords[:, 0], 0, H-1)
        pred_coords[:, 1] = np.clip(pred_coords[:, 1], 0, W-1)
        
        # Reshape based on input
        if is_trajectory:
            pred_coords = pred_coords.reshape(B, T, 2)
            return Trajectory(coord=pred_coords)
        else:
            return AgentState(coord=pred_coords)

    def get_response(
            self, 
            agent_data: Union[AgentState, Trajectory],
            return_format='dict', 
            keys=None, 
            str_filter=None, 
            type_filter=None
        ):
        """
        Get sensory responses for the given trajectory.

        Parameters:
            - traj: rtgym.dataclass.Trajectory object
            - return_format: format of the returned responses. Can be 'dict' or 'array'.
            - keys: list of sensory keys to get responses. If None, get responses for all.
            - str_filter: filter the sensory keys by the given string.
            - type_filter: filter the sensory keys by the given type.

        Returns:
            - responses: sensory responses. The responses are of shape (n_cells, *arena_dimensions).
                         After indexing, it will be of shape (n_cells, n_batch).
                         When return_format is 'dict', it will be a dictionary of responses.
                         When return_format is 'array', it will be a numpy array of responses.
        """
        # Set filter_keys if not provided
        keys = self.filter_sensories(keys, str_filter, type_filter)
        if return_format == 'dict':
            return {key: self.sensories[key].get_response(agent_data) for key in keys}
        elif return_format == 'array':
            res_list = [self.sensories[key].get_response(agent_data) for key in keys]
            return np.concatenate(res_list, axis=-1)
        else:
            raise ValueError(f"Unknown return format: {return_format}")

    def compute_res(self):
        for _sens in self.sensories.values():
            _sens._compute_res()

    def save(self, file_path):
        """
        Save the sensory cells to a file.

        Parameters:
            file_path: Path to the file where the sensory cells will be saved.
        """
        all_sensory_data = {}
        for key, _sens in self.sensories.items():
            all_sensory_data[key] = _sens.state_dict()
        with open(file_path, 'wb') as f:
            pickle.dump(all_sensory_data, f)

    def load_from_state_dict(self, state_dict, append=True):
        """
        Load the sensory cells from a state dictionary.

        Parameters:
            state_dict: state dictionary of the sensory cells.
            append: if True, append the sensory cells to the existing sensory cells.
                    if False, replace the existing sensory cells.
        """
        if not append:
            self.sensories = {}
        for key in state_dict.keys():
            _sens_data = state_dict[key]
            _sens_class = self._get_sensory_class(_sens_data.pop('sens_type'))
            _sens = _sens_class.load_from_dict(_sens_data, self.arena)
            self.sensories[key] = _sens

    def load(self, file_path):
        """
        Load the sensory cells from a file.

        Parameters:
            file_path: Path to the file where the sensory cells are saved.
        """
        with open(file_path, 'rb') as f:
            all_sensory_data = pickle.load(f)
        self.load_from_state_dict(all_sensory_data)
