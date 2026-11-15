"""Train the HC-MEC model and build the ratemaps used by the demo notebook."""

from __future__ import annotations

import argparse
import gc
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from remi.model import HCMEC
from remi.rtgym_lite.utils.data_processing import RatemapAggregator


DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HC-MEC and save its checkpoint and ratemaps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=Path(".demo_artifacts"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    training = parser.add_argument_group("HC-MEC training")
    training.add_argument("--epochs", type=int, default=50)
    training.add_argument("--steps-per-epoch", type=int, default=1_000)
    training.add_argument("--batch-size", type=int, default=128)
    training.add_argument("--trajectory-seconds", type=float, default=2.0)
    training.add_argument("--learning-rate", type=float, default=1e-3)
    training.add_argument("--gradient-clip", type=float, default=0.1)
    training.add_argument("--max-masking-ratio", type=float, default=1.0)

    ratemaps = parser.add_argument_group("ratemap collection")
    ratemaps.add_argument("--ratemap-batches", type=int, default=200)
    ratemaps.add_argument("--ratemap-batch-size", type=int, default=512)
    ratemaps.add_argument("--ratemap-seconds", type=float, default=5.0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA-capable GPU is available.")
    return torch.device(name)


def make_gym_config():
    return OmegaConf.create(
        {
            "temporal_resolution": 50,
            "spatial_resolution": 1,
            "arena_cfg": {
                "arena_shape": "rectangle",
                "arena_params": {"dimensions": [100, 100]},
            },
        }
    )


def make_hcmec_config(max_masking_ratio: float):
    return OmegaConf.create(
        {
            "behavior": {
                "avoid_boundary_dist": 10,
                "max_velocity": 10,
                "random_drift_magnitude": 0.05,
                "switch_direction_prob": 0.15,
                "switch_velocity_prob": 0.02,
                "velocity_mean": 10,
                "velocity_sd": 10,
            },
            "modules": {
                "hpc": {
                    "hidden_dim": 512,
                    "alpha": 0.2,
                    "learn_alpha": True,
                    "weight": "learn",
                },
                "wsm": {
                    "alpha": 0.2,
                    "learn_alpha": True,
                    "weight": "learn",
                    "sensory_profile": {
                        "wsm": {
                            "type": "weak_sm_cell",
                            "sigma": 12,
                            "ssigma": 3,
                            "n_cells": 256,
                            "magnitude": 3,
                            "normalize": True,
                        }
                    },
                    "data_cfg": {
                        "init": "masked_init",
                        "input": {"keys": ["wsm"]},
                        "label": {"keys": ["wsm"]},
                    },
                    "masking_prob": max_masking_ratio,
                    "masking_method": "random",
                    "loss_cfg": {"mse_norm": 10},
                },
                "gc": {
                    "alpha": 0.2,
                    "learn_alpha": True,
                    "weight": "learn",
                    "sensory_profile": {
                        "gc": {
                            "type": "grid_cell",
                            "sigma2scale": 3.26,
                            "initial_scale": 30,
                            "n_scales": 6,
                            "magnitude": 1.12,
                            "normalize": True,
                            "n_cells_per_group": 48,
                            "n_groups_per_scale": 1,
                            "theta": 60,
                        }
                    },
                    "data_cfg": {
                        "init": "masked_init",
                        "input": {"keys": []},
                        "label": {"keys": ["gc"]},
                    },
                    "masking_prob": max_masking_ratio,
                    "masking_method": "random",
                    "loss_cfg": {"mse_norm": 10},
                },
                "dir": {
                    "alpha": 0.2,
                    "learn_alpha": True,
                    "weight": "learn",
                    "sensory_profile": {
                        "dir": {
                            "type": "direction_cell",
                            "n_cells": 32,
                            "magnitude": 5,
                            "normalize": True,
                            "sigma": 1,
                            "n_bins": 360,
                        }
                    },
                    "data_cfg": {
                        "init": "masked_init",
                        "input": {"keys": ["dir"]},
                        "label": {"keys": ["dir"]},
                    },
                    "masking_prob": max_masking_ratio,
                    "masking_method": "random",
                    "loss_cfg": {"mse_norm": 1},
                },
                "vel": {
                    "alpha": 0.2,
                    "learn_alpha": True,
                    "weight": "learn",
                    "sensory_profile": {
                        "vel": {
                            "type": "velocity_abs",
                            "n_cells": 32,
                            "magnitude": 200,
                            "msigma": 50,
                        }
                    },
                    "data_cfg": {
                        "init": "masked_init",
                        "input": {"keys": ["vel"]},
                        "label": {"keys": ["vel"]},
                    },
                    "masking_prob": max_masking_ratio,
                    "masking_method": "random",
                    "loss_cfg": {"mse_norm": 1},
                },
            },
        }
    )


def build_model(args: argparse.Namespace, device: torch.device) -> HCMEC:
    model = HCMEC(
        make_hcmec_config(args.max_masking_ratio),
        rep_seed=args.seed,
        room_idx=0,
    )
    model.set_gym(make_gym_config())
    model.to(device)

    assert model.size == 1_120
    assert model.get_region("gc").input_dimension == 0
    return model


def train_hcmec(
    model: HCMEC,
    args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, list[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_history: list[float] = []
    total_steps = args.epochs * args.steps_per_epoch
    model.train()

    for step in range(total_steps):
        initial_state, inputs, labels, _ = model.generate_data(
            duration=args.trajectory_seconds,
            batch_size=args.batch_size,
        )
        _, states = model(inputs, initial_state)
        loss, _, _ = model.compute_loss(states, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        loss_history.append(float(loss.detach()))

        if (step + 1) % 100 == 0:
            gc.collect()
        if (step + 1) % args.steps_per_epoch == 0:
            recent_loss = np.mean(loss_history[-args.steps_per_epoch :])
            epoch = (step + 1) // args.steps_per_epoch
            print(f"epoch {epoch:3d}/{args.epochs}: loss={recent_loss:.3f}")

    return optimizer, loss_history


def save_checkpoint(
    path: Path,
    model: HCMEC,
    optimizer: torch.optim.Optimizer,
    loss_history: list[float],
    args: argparse.Namespace,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": args.epochs,
            "loss_history": np.asarray(loss_history, dtype=np.float32),
            "training_args": vars(args),
        },
        path,
    )


def collect_ratemaps(model: HCMEC, args: argparse.Namespace) -> dict[str, np.ndarray]:
    learned = {
        key: RatemapAggregator(model.gym.arena.arena_map)
        for key in ("wsm", "gc", "hpc")
    }
    targets = {
        key: RatemapAggregator(model.gym.arena.arena_map)
        for key in ("wsm", "gc")
    }

    # Training mode keeps the same masking behavior used for the reported maps.
    model.train()
    with torch.no_grad():
        for batch in range(args.ratemap_batches):
            initial_state, inputs, labels, coordinates = model.generate_data(
                duration=args.ratemap_seconds,
                batch_size=args.ratemap_batch_size,
            )
            _, states = model(inputs, initial_state)

            for key, aggregator in learned.items():
                aggregator.update(states[model.region_dict[key]], coordinates)
            for key, aggregator in targets.items():
                aggregator.update(labels[model.region_dict[key]], coordinates)

            if (batch + 1) % 10 == 0:
                print(f"ratemaps {batch + 1:3d}/{args.ratemap_batches}")
                gc.collect()
                if model.device.type == "cuda":
                    torch.cuda.empty_cache()

    model.eval()
    arrays = {
        "arena_map": np.asarray(model.gym.arena.arena_map),
        "visit_counts": learned["gc"].visit_counts.cpu().numpy(),
    }
    arrays.update(
        {f"learned_{key}": value.get_ratemap().cpu().numpy() for key, value in learned.items()}
    )
    arrays.update(
        {f"target_{key}": value.get_ratemap().cpu().numpy() for key, value in targets.items()}
    )
    return arrays


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    print(f"device: {device}")
    model = build_model(args, device)
    print(f"hidden units: {model.size}")

    optimizer, loss_history = train_hcmec(model, args)
    checkpoint_path = args.output_dir / "hcmec.pt"
    save_checkpoint(checkpoint_path, model, optimizer, loss_history, args)
    print(f"saved {checkpoint_path}")

    ratemap_arrays = collect_ratemaps(model, args)
    ratemap_path = args.output_dir / "ratemaps.npz"
    np.savez_compressed(ratemap_path, **ratemap_arrays)
    print(f"saved {ratemap_path}")


if __name__ == "__main__":
    main()
