from termcolor import colored
from omegaconf import OmegaConf
from datetime import datetime
import argparse
import numpy as np
import torch

def printc(msg, c='g', border=None, end='\n'):
    """
    Print a message with a color.

    **Parameters:**
        - `msg` (str): The message to print.
        - `c` (str): The color to print the message in.
        - `border` (str): The border to print around the message.
        - `end` (str): The end of the message.
    """
    # map abbreviated color to full color (such as 'g', 'r', 'b', 'y', 'm', 'c', 'o')
    if c in ['g', 'r', 'b', 'y', 'm', 'c', 'o']:
        color = {
            'g': 'green',
            'r': 'red',
            'b': 'blue',
            'y': 'yellow',
            'm': 'magenta',
            'c': 'cyan',
            'o': 'orange'
        }[c]
    if border is not None:
        print(colored('='*50, color), end=end)
    print(colored(msg, color), end=end)
    if border is not None:
        print(colored('='*50, color), end=end)


def print_cfg(msg, cfg):
    """
    Print a message with a color and a configuration.

    **Parameters:**
        - `msg` (str): The message to print.
        - `cfg` (OmegaConf): The configuration to print.
    """
    printc(msg, 'c')
    print("  " + OmegaConf.to_yaml(cfg).replace('\n', '\n  '))


def print_cfg_dict(msg, cfg_dict, space=True, max_level=None):
    """
    Print a message with a color and a configuration dictionary.

    **Parameters:**
        - `msg` (str): The message to print.
        - `cfg_dict` (dict): The configuration dictionary to print.
        - `space` (bool): Whether to print a space after the message.
        - `max_level` (int): The maximum level of the configuration dictionary to print.
    """
    printc(msg, 'c')
    
    def print_dict(d, level=1):
        # If a max_level is specified, stop printing deeper levels.
        if max_level is not None and level > max_level:
            return
        indent = '  ' * level
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{indent}{key}:")
                print_dict(value, level + 1)
            elif isinstance(value, (np.ndarray, torch.Tensor)):
                print(f"{indent}{key}: {type(value).__name__} with shape {value.shape}")
            else:
                print(f"{indent}{key}: {value}")

    print_dict(cfg_dict)
    if space:
        print()  # Print a blank line afterwards


def time_str():
    """
    Get the current time as a string.

    **Returns:**
        - `str`: The current time as a string.
    """
    return datetime.now().strftime("%y%m%d_%H%M%S")


def print_args(args: argparse.Namespace):
    """
    Print the arguments.

    **Parameters:**
        - `args` (argparse.Namespace): The arguments to print.
    """
    printc("="*50, 'c')
    print_cfg_dict("Args", vars(args), space=False)
    printc("="*50, 'c')
