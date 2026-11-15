import torch


INIT_FNS = {
    "xavier_uniform": torch.nn.init.xavier_uniform_,
    "xavier_normal":  torch.nn.init.xavier_normal_,
    "kaiming_uniform": torch.nn.init.kaiming_uniform_,
    "kaiming_normal": torch.nn.init.kaiming_normal_,
    "zeros": torch.nn.init.zeros_,
    "ones": torch.nn.init.ones_,
    "uniform": torch.nn.init.uniform_,
    "normal": torch.nn.init.normal_,
}


def get_init_fn(init_type: str) -> callable:
    """
    Get the initialization function for the given initialization type
    """
    if init_type not in INIT_FNS:
        raise ValueError(f"Invalid initialization type: {init_type}")
    return INIT_FNS[init_type]


def apply_initialization(param: torch.Tensor, init_type: str, **kwargs):
    """
    Apply initialization to a parameter with correct arguments for each init type.
    
    Args:
        param: The tensor parameter to initialize
        init_type: The initialization method name
        **kwargs: Additional arguments (e.g., gain, a, mode, nonlinearity)
    """
    init_fn = get_init_fn(init_type)
    
    # Handle different initialization functions with their specific parameters
    if init_type in ["xavier_uniform", "xavier_normal"]:
        # Xavier methods accept gain parameter
        gain = kwargs.get("gain", 1.0)
        init_fn(param, gain=gain)
    elif init_type in ["kaiming_uniform", "kaiming_normal"]:
        # Kaiming methods accept a, mode, and nonlinearity parameters
        a = kwargs.get("a", 0)
        mode = kwargs.get("mode", "fan_in")
        nonlinearity = kwargs.get("nonlinearity", "leaky_relu")
        init_fn(param, a=a, mode=mode, nonlinearity=nonlinearity)
    elif init_type == "uniform":
        # Uniform distribution with bounds
        a = kwargs.get("a", 0.0)
        b = kwargs.get("b", 1.0)
        init_fn(param, a=a, b=b)
    elif init_type == "normal":
        # Normal distribution with mean and std
        mean = kwargs.get("mean", 0.0)
        std = kwargs.get("std", 1.0)
        init_fn(param, mean=mean, std=std)
    elif init_type in ["zeros", "ones"]:
        # These functions only accept the tensor parameter
        init_fn(param)
    else:
        # Fallback for any other initialization functions
        init_fn(param)
