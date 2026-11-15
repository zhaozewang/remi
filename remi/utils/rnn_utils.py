import torch
from remi.nn4n.nn import (
    LinearLayer, 
    BlockRecurrentLayer, 
    RecurrentLayer, 
    LeakyLinearLayer
)


def construct_recurrent_block(input_dim, n_hidden, region_cfg):
    """
    Construct a recurrent block.

    **Parameters:**
        - `input_dim` (int): The input dimension.
        - `n_hidden` (int): The number of hidden units.
        - `region_cfg` (dict): The configuration for the region.

    **Returns:**
        - `RecurrentLayer`: The recurrent block.
    """
    projection_layer = None
    if input_dim is not None:
        projection_layer = LinearLayer(input_dim=input_dim, output_dim=n_hidden)

    return RecurrentLayer(
        projection_layer=projection_layer,
        leaky_layer=LeakyLinearLayer(
            linear_layer=LinearLayer(input_dim=n_hidden, output_dim=n_hidden),
            activation=torch.nn.ReLU(),
            alpha=region_cfg["alpha"],
            learn_alpha=region_cfg["learn_alpha"]
        )
    )


def set_between_block_projections(block_recurrent_layer, override=False):
    """
    Set the between-block projections.

    **Parameters:**
        - `block_recurrent_layer` (BlockRecurrentLayer): The block recurrent layer.
        - `override` (bool): Whether to override the existing projections.
    """
    block_sizes = block_recurrent_layer.block_sizes()
    for i in range(len(block_sizes)):
        for j in range(len(block_sizes)):
            if i == j:
                continue
            if not override and block_recurrent_layer.get_projection(from_idx=i, to_idx=j) is not None:
                continue
            block_recurrent_layer.set_projection(from_idx=i, to_idx=j, projection_layer=LinearLayer(
                input_dim=block_sizes[i], output_dim=block_sizes[j])
            )
            block_recurrent_layer.set_projection(from_idx=j, to_idx=i, projection_layer=LinearLayer(
                input_dim=block_sizes[j], output_dim=block_sizes[i])
            )


def init_block_with_subnet(subnet, subnet_region_dict, n_add_blocks):
    """
    Initialize a block recurrent layer with a subnet.

    **Parameters:**
        - `subnet` (nn.Module): The subnet to initialize the block recurrent layer with.
        - `subnet_region_dict` (dict): The region dictionary for the subnet.
        - `n_add_blocks` (int): The number of additional blocks to add.

    **Returns:**
        - `BlockRecurrentLayer`: The block recurrent layer.
        - `dict`: The new region dictionary.
    """
    block_recurrent_layer = BlockRecurrentLayer(n_blocks=subnet.n_blocks+n_add_blocks)

    subnet.requires_grad = False
    new_region_dict = {}
    for key, i in subnet_region_dict.items():
        for j in subnet_region_dict.values():
            if i == j:
                new_region_dict[key] = i
                block_recurrent_layer.set_recurrent(i, subnet.get_recurrent(i))
            else:
                block_recurrent_layer.set_projection(
                    from_idx=i, to_idx=j, 
                    projection_layer=subnet.get_projection(i, j)
                )

    # Return the block recurrent layer and the new region dict
    return block_recurrent_layer, new_region_dict
