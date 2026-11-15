"""Small plotting helpers used by the REMI demo notebook."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch


REGION_STYLES = {
    "wsm": {"label": "SMC", "color": "#ef4444", "marker": "o"},
    "gc": {"label": "GC", "color": "#16a34a", "marker": "^"},
    "hpc": {"label": "HPC", "color": "#f59e0b", "marker": "*"},
}


def configure_plot_style() -> None:
    """Use a white, minimal plotting style throughout the notebook."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def spatial_information(rate_map: np.ndarray, bin_size: int = 20) -> float:
    """Estimate the spatial information of one firing-rate map.

    The map is normalized to ``[0, 1]`` and spatially averaged into square bins
    before information is computed. This score is used only to select clear HPC
    examples for display.
    """
    normalized = (rate_map - np.nanmin(rate_map)) / (
        np.nanmax(rate_map) - np.nanmin(rate_map) + 1e-12
    )
    binned = np.array(
        [
            np.nanmean(normalized[row : row + bin_size, col : col + bin_size])
            for row in range(0, normalized.shape[0], bin_size)
            for col in range(0, normalized.shape[1], bin_size)
        ]
    )

    mean_rate = np.nanmean(binned)
    positive = binned[(binned > 0) & np.isfinite(binned)]
    probabilities = positive / np.nansum(binned)
    return float(np.sum(probabilities * (positive / mean_rate) * np.log2(positive / mean_rate)))


def select_hpc_units(
    hpc_maps: np.ndarray,
    count: int = 6,
    min_mean_rate: float = 0.01,
    min_information: float = 20.0,
) -> np.ndarray:
    """Select active, spatially informative HPC units for visualization."""
    mean_rates = np.nanmean(hpc_maps, axis=(1, 2))
    information = np.array([spatial_information(rate_map) for rate_map in hpc_maps])
    candidates = np.flatnonzero(
        (mean_rates > min_mean_rate) & (information > min_information)
    )

    if len(candidates) < count:
        candidates = np.flatnonzero(mean_rates > min_mean_rate)
    ranked = candidates[np.argsort(mean_rates[candidates])[::-1]]
    return ranked[:count]


def plot_ratemaps(
    learned: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
) -> None:
    """Plot representative target and learned SMC, GC, and HPC rate maps."""
    smc_indices = np.arange(6)
    gc_indices = np.arange(6) * 48
    hpc_indices = select_hpc_units(learned["hpc"])
    rows = (
        ("SMC target", targets["wsm"], smc_indices),
        ("SMC learned", learned["wsm"], smc_indices),
        ("GC target", targets["gc"], gc_indices),
        ("GC learned", learned["gc"], gc_indices),
        ("HPC learned", learned["hpc"], hpc_indices),
    )

    color_map = plt.colormaps["jet"].copy()
    color_map.set_bad("white")
    fig, axes = plt.subplots(
        len(rows),
        6,
        figsize=(13, 11),
        constrained_layout=True,
        facecolor="white",
    )

    for row, (label, maps, indices) in enumerate(rows):
        for column, unit in enumerate(indices):
            axes[row, column].imshow(maps[unit], cmap=color_map)
            axes[row, column].set_title(f"#{unit}", fontsize=9)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(label, fontsize=11, weight="bold", labelpad=12)

    plt.show()


def plot_recall_error(
    recall_distances: Mapping[str, np.ndarray],
    time_seconds: np.ndarray,
) -> None:
    """Plot mean recall error with the standard error across query locations."""
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    query_count = next(iter(recall_distances.values())).shape[0]

    for key in ("wsm", "gc", "hpc"):
        distances = recall_distances[key]
        mean = distances.mean(axis=0)
        sem = distances.std(axis=0) / np.sqrt(query_count)
        style = REGION_STYLES[key]
        ax.plot(time_seconds, mean, lw=2, color=style["color"], label=style["label"])
        ax.fill_between(
            time_seconds,
            mean - sem,
            mean + sem,
            color=style["color"],
            alpha=0.14,
            linewidth=0,
        )

    ax.set_xlabel("query time (s)")
    ax.set_ylabel("distance to cue location (cm)")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def select_recall_trial(
    recall_distances: Mapping[str, np.ndarray],
    final_threshold: float = 7.0,
) -> int:
    """Choose the first jointly successful trial, or the best available trial."""
    successful = np.flatnonzero(
        (recall_distances["wsm"][:, -1] < final_threshold)
        & (recall_distances["gc"][:, -1] < final_threshold)
        & (recall_distances["hpc"][:, -1] < final_threshold)
    )
    if len(successful):
        return int(successful[0])

    final_error = sum(
        recall_distances[key][:, -1] for key in ("wsm", "gc", "hpc")
    )
    return int(np.argmin(final_error))


def plot_recall_snapshots(
    arena_map: np.ndarray,
    query_positions: np.ndarray,
    recall_coordinates: Mapping[str, np.ndarray],
    recall_distances: Mapping[str, np.ndarray],
    time_seconds: np.ndarray,
    trial: int,
) -> None:
    """Show decoded recall positions as the network converges on one cue."""
    jointly_close = (
        (recall_distances["wsm"][trial] < 15)
        & (recall_distances["gc"][trial] < 15)
    )
    first_close = (
        int(np.flatnonzero(jointly_close)[0])
        if jointly_close.any()
        else len(time_seconds) - 1
    )
    steps = np.linspace(0, max(first_close, 3), 4, dtype=int)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(14, 3.5),
        constrained_layout=True,
        facecolor="white",
    )
    query_y, query_x = query_positions[trial]

    for ax, step in zip(axes, steps):
        ax.imshow(arena_map, cmap="gray_r", vmin=0, vmax=1)
        ax.plot([query_x - 4, query_x + 4], [query_y, query_y], color="#2563eb", lw=3)
        ax.plot([query_x, query_x], [query_y - 4, query_y + 4], color="#2563eb", lw=3)

        for key, size in (("wsm", 70), ("gc", 80), ("hpc", 120)):
            decoded_y, decoded_x = recall_coordinates[key][trial, step]
            style = REGION_STYLES[key]
            ax.scatter(
                decoded_x,
                decoded_y,
                marker=style["marker"],
                color=style["color"],
                s=size,
                alpha=0.85,
                edgecolors="none",
            )

        ax.set_title(f"{time_seconds[step]:.2f} s")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def _as_numpy(value) -> np.ndarray:
    """Detach a tensor when necessary and return a NumPy array."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def plot_recall_manifolds(
    learned_ratemaps: Mapping[str, np.ndarray],
    recalled_states,
    region_indices: Mapping[str, int],
    query_positions: np.ndarray,
    trial: int,
    seed: int = 42,
) -> None:
    """Project each population manifold and one recall trajectory into 3D PCA space."""
    fig = plt.figure(figsize=(14, 4.5), constrained_layout=True, facecolor="white")

    for panel, key in enumerate(("wsm", "gc", "hpc"), start=1):
        rate_map = learned_ratemaps[key]
        valid = np.isfinite(rate_map).all(axis=0)
        manifold = np.moveaxis(rate_map, 0, -1)[valid]
        reducer = PCA(
            n_components=3,
            svd_solver="randomized",
            random_state=seed,
        ).fit(manifold)

        manifold_3d = reducer.transform(manifold[::8])
        response = _as_numpy(recalled_states[region_indices[key]][trial])
        response_3d = reducer.transform(response)
        row, column = np.rint(query_positions[trial]).astype(int)
        goal_3d = reducer.transform(rate_map[:, row, column][None])

        ax = fig.add_subplot(1, 3, panel, projection="3d", facecolor="white")
        ax.scatter(*manifold_3d.T, s=2, color="#2563eb", alpha=0.06)
        ax.plot(*response_3d.T, color="black", lw=1, alpha=0.45)
        ax.scatter(
            *response_3d.T,
            c=np.linspace(0, 1, len(response_3d)),
            cmap="viridis",
            s=10,
        )
        ax.scatter(*goal_3d.T, color="#16a34a", s=85)
        ax.set_title(REGION_STYLES[key]["label"], weight="bold", pad=8)
        ax.set_axis_off()

    plt.show()


def plot_training_loss(loss_history: list[float], title: str) -> None:
    """Plot a compact loss curve for a training loop."""
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="white")
    ax.plot(loss_history, color="#2563eb", lw=1)
    ax.set_xlabel("training batch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_planning_paths(
    reference_paths: np.ndarray,
    decoded_paths: Mapping[str, np.ndarray],
    examples: int = 4,
) -> None:
    """Compare reference paths with decoded GC plans and SMC reconstructions."""
    columns = 2
    rows = int(np.ceil(examples / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(10, 5 * rows),
        constrained_layout=True,
        facecolor="white",
    )

    for trial, ax in enumerate(np.asarray(axes).flat[:examples]):
        reference = reference_paths[trial]
        gc_path = decoded_paths["gc"][trial]
        smc_path = decoded_paths["wsm"][trial]
        ax.plot(reference[:, 1], reference[:, 0], color="0.82", lw=2, label="reference")
        ax.plot(gc_path[:, 1], gc_path[:, 0], color="#2563eb", lw=2, label="GC plan")
        ax.plot(smc_path[:, 1], smc_path[:, 0], color="#ef4444", lw=2, label="SMC reconstruction")
        ax.scatter(reference[0, 1], reference[0, 0], marker="x", s=80, color="black")
        ax.scatter(reference[-1, 1], reference[-1, 0], marker="*", s=120, color="#16a34a")
        ax.set(aspect="equal", xlim=(0, 109), ylim=(109, 0))

    axes.flat[0].legend(frameon=False, fontsize=8)
    plt.show()
