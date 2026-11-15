import numpy as np
import torch
import torch.nn.functional as F


class Masking:
    """
    Base class for all masking utilities.
    """
    def __init__(
            self,
            device: torch.device = torch.device('cpu'),
            **kwargs
        ):
        self.device = device

    def to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.float().to(self.device)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(self.device)
        else:
            raise ValueError(f"Unsupported input type: {type(x)}")

    def __call__(self, x, mr=None, sigma_t=None):
        return self.mask(x, mr, sigma_t)

    def to(self, device: torch.device):
        self.device = device
        return self

    def gaussian_kernel_1d(self, kernel_size: int, sigma: float):
        """Creates a 1D Gaussian kernel."""
        x = torch.arange(kernel_size) - kernel_size // 2
        gauss = torch.exp(-x.float()**2 / (2 * sigma**2))
        gauss /= gauss.sum()
        return gauss

    def apply_gaussian_blur(self, mask: torch.Tensor, sigma_t=None):
        """
        Apply Gaussian blur to mask tensor with option for per-batch sigma values.
        
        **Parameters:**
        - mask: Tensor of shape (batch_size, T, D)
        - sigma_t: Optional tensor of shape (batch_size,) with per-batch sigma values
                   If None, no temporal smoothing is applied
        
        **Returns:**
        - Blurred mask tensor if sigma_t is provided, otherwise original mask
        """
        # If no sigma values provided, return original mask (no smoothing)
        if sigma_t is None or (isinstance(sigma_t, torch.Tensor) and (sigma_t <= 0).all()):
            return mask

        batch_size = mask.shape[0]
        
        # Process each batch element with its own sigma
        processed_batches = []
        for i in range(batch_size):
            # Skip smoothing if sigma is zero or negative
            curr_sigma = sigma_t[i].item() if hasattr(sigma_t[i], 'item') else sigma_t[i]
            if curr_sigma <= 0:
                processed_batches.append(mask[i:i+1])
                continue
                
            # Calculate kernel size based on sigma
            k_size = max(int(6 * curr_sigma) | 1, 3)
            
            # Create and apply kernel
            kernel = self.gaussian_kernel_1d(k_size, curr_sigma).to(mask.device)
            kernel = kernel.view(1, 1, -1, 1)
            
            # Pad and apply convolution
            pad = k_size // 2
            m = mask[i:i+1].unsqueeze(1)  # Shape: (1, 1, T, D)
            m = F.pad(m, (0, 0, pad, pad), mode='reflect')
            blurred = F.conv2d(m, kernel, padding=0).squeeze(1)  # Shape: (1, T, D)
            processed_batches.append(blurred)
        
        # Combine processed batches
        mask = torch.cat(processed_batches, dim=0)
        return mask

    def new_mask(self, x, mr=None, sigma_t=None):
        """
        Generates a mask with different masking thresholds for each dimension.
        The thresholds are determined based on per-batch masking ratios.
        
        **Args:**
        - x (torch.Tensor): Input tensor of shape (batch_size, T, D)
        - mr (torch.Tensor): Masking ratio per batch (batch_size,)
        - sigma_t (torch.Tensor, optional): Smoothing sigma per batch (batch_size,)

        **Returns:**
        - torch.Tensor: Mask tensor of the same shape as x
        """
        # Generate a random mask
        mask = torch.rand(x.shape, device=self.device)
        
        # Apply temporal smoothing if sigma_t is provided
        if sigma_t is not None:
            mask = self.apply_gaussian_blur(mask, sigma_t)  # Shape: (batch_size, T, D)        

        # Generate masking ratios for each dimension
        sorted_mask, _ = mask.view(x.shape[0], -1).sort(dim=1)
        max_size = sorted_mask.size(1) - 1
        q_indices = (mr.unsqueeze(1) * (max_size)).long().clamp(max=max_size)
        thresholds = sorted_mask.gather(dim=1, index=q_indices).unsqueeze(1)
        mask = mask > thresholds

        return mask.float()

    def mask(self, x, mr=None, sigma_t=None):
        """
        **Args:**
        - x (torch.Tensor): Input tensor of shape (batch_size, T, D)
        - mr (torch.Tensor, optional): Masking ratio per batch (batch_size,)
        - sigma_t (torch.Tensor, optional): Smoothing sigma per batch (batch_size,)
                    If None, no temporal smoothing is applied
        
        **Returns:**
        - torch.Tensor: Masked tensor with the same shape as x
        """
        x = self.to_tensor(x)  # Ensure tensor
        if mr is not None:
            mask = self.new_mask(x, mr, sigma_t)  # Generate mask
            return x * mask  # Apply mask
        else:
            return x
