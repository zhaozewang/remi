import torch
from torch import Tensor
from typing import Optional, Union
from ..utils import apply_initialization


class LinearLayer(torch.nn.Module):
    r"""A fully-connected layer with optional sparsity and positivity masks.

    A linear transformation with optional constraints on the weights. The transformation
    is defined as :math:`y = xA^T + b` where A is the weight matrix and b is the bias.
    The weights can be constrained by sparsity (forcing certain weights to zero) and
    positivity (forcing certain weights to be positive or negative only).

    The sparsity mask is a binary tensor that determines which weights are allowed to be
    non-zero (1) and which must be zero (0). The positivity mask is a ternary tensor
    that determines which weights must be non-negative (1), non-positive (-1), or
    unconstrained (0).

    Args:
        input_dim (int): size of each input sample
        output_dim (int): size of each output sample
        weight_init (str, optional): weight initialization method (default: "xavier_uniform")
        bias_init (str, optional): bias initialization method (default: "zeros")
        sparsity_mask (Tensor, optional): binary mask for weight sparsity (default: None)
        positivity_mask (Tensor, optional): ternary mask for weight signs (default: None)
        dtype (torch.dtype, optional): data type of parameters (default: torch.float32)
        device (Union[str, torch.device], optional): device of parameters (default: "cpu")

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{input\_dim}`
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{output\_dim}`

    Examples::

        >>> # Basic usage
        >>> m = LinearLayer(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])

        >>> # With sparsity mask (only allow connections between even indices)
        >>> sparsity = torch.zeros(30, 20)
        >>> sparsity[::2, ::2] = 1
        >>> m = LinearLayer(20, 30, sparsity_mask=sparsity)
        >>> output = m(input)
        >>> print(m.weight[1, 1])  # Masked weight is zero
        tensor(0.)

        >>> # With positivity mask (excitatory and inhibitory neurons)
        >>> positivity = torch.zeros(30, 20)
        >>> positivity[:15] = 1  # First half excitatory
        >>> positivity[15:] = -1  # Second half inhibitory
        >>> m = LinearLayer(20, 30, positivity_mask=positivity)
        >>> print((m.weight[:15] >= 0).all())  # First half all positive
        tensor(True)
        >>> print((m.weight[15:] <= 0).all())  # Second half all negative
        tensor(True)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        weight_init: str = "xavier_uniform",
        bias_init: str = "zeros",
        sparsity_mask: Optional[Tensor] = None,
        positivity_mask: Optional[Tensor] = None,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim

        self.weight = torch.nn.Parameter(
            torch.empty(output_dim, input_dim, dtype=dtype, device=device)
        )
        self._apply_init(self.weight, weight_init)
        if bias_init is None:
            self.register_parameter("bias", None)
        else:
            self.bias = torch.nn.Parameter(
                torch.empty(output_dim, dtype=dtype, device=device)
            )
            self._apply_init(self.bias, bias_init)

        self.sparsity_mask = self._register_mask(sparsity_mask, "sparsity")
        self.positivity_mask = self._register_mask(positivity_mask, "positivity")

        self.register_forward_pre_hook(self._enforce_constraints)
        self._register_backward_hooks()

    # =========================================================================
    # METHODS
    # =========================================================================
    def _enforce_constraints(self, *_):
        """Apply sparsity and positivity masks just before every forward."""
        if self.sparsity_mask is not None:
            self.weight.data.mul_(self.sparsity_mask)

        if self.positivity_mask is not None:
            w = self.weight.data
            pos = self.positivity_mask == 1
            neg = self.positivity_mask == -1
            w[pos] = torch.clamp(w[pos], min=0)
            w[neg] = torch.clamp(w[neg], max=0)

    def _register_backward_hooks(self):
        """Zero out gradients for pruned weights."""
        if self.sparsity_mask is not None:
            self.weight.register_hook(
                lambda grad: grad * self.sparsity_mask.float()
            )

    def _register_mask(self, mask: Optional[Tensor], name: str) -> Optional[Tensor]:
        """Register a mask as a buffer (does not require grad but will be saved)"""
        if mask is not None:
            if mask.shape != self.weight.shape:
                raise ValueError(f"{name}_mask shape {mask.shape} ≠ weight shape {self.weight.shape}")
            self.register_buffer(f"{name}_mask", mask.bool())
        else:
            self.register_buffer(f"{name}_mask", None)
        return getattr(self, f"{name}_mask")

    def _apply_init(self, param: Tensor, init_type: str, **kwargs):
        """Apply initialization to a parameter with correct arguments"""
        apply_initialization(param, init_type, **kwargs)

    def skeletonize(self):
        """
        Clear the parameters of the layer
        """
        del self.weight, self.bias
        if hasattr(self, "sparsity_mask"):
            del self.sparsity_mask
        if hasattr(self, "positivity_mask"):
            del self.positivity_mask

    def freeze(self):
        """Disable gradients for the layer parameters."""
        self.requires_grad_(False)

    def unfreeze(self):
        """Enable gradients for the layer parameters."""
        self.requires_grad_(True)

    # TRAINING
    # =========================================================================
    def forward(self, x: Tensor) -> Tensor:
        y = x.float() @ self.weight.T
        return y if self.bias is None else y + self.bias
