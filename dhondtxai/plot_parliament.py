import math
import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def plot_parliament(
    total_seats,
    features,
    seats,
    slices=50,
    additional_rows=5,
    colors=None,
    title=None,
    show=True,
):
    """Plot a semicircular parliament view.

    This keeps the original DhondtXAI visualization API while allowing callers
    to pass explicit colors for positive/negative signed explanations.
    """
    features = list(features)
    seats = np.asarray(seats, dtype=int)

    if len(features) != len(seats):
        raise ValueError("Number of features and seats must match.")
    if total_seats < int(seats.sum()):
        total_seats = int(seats.sum())
    if total_seats <= 0:
        raise ValueError("total_seats must be positive.")

    if colors is None:
        colors = _default_colors(len(features))
    else:
        colors = list(colors)
        if len(colors) != len(features):
            raise ValueError("Number of colors must match number of features.")

    feature_colors = {feature: colors[i % len(colors)] for i, feature in enumerate(features)}
    sorted_features = sorted(features, key=lambda feature: seats[features.index(feature)], reverse=True)

    pieces_per_slice_without_additional = max(1, math.ceil(total_seats / slices))
    pieces_per_slice = pieces_per_slice_without_additional + additional_rows
    angle_gap = 0.2
    radial_angles = np.linspace(180, 0, slices + 1)

    fig, ax = plt.subplots(figsize=(14, 8))
    radius = 10
    piece_depth = radius / pieces_per_slice
    start_radius = piece_depth * additional_rows

    current_feature = 0
    remaining_seats = seats[features.index(sorted_features[current_feature])] if sorted_features else 0
    current_color = feature_colors[sorted_features[current_feature]] if sorted_features else "tab:gray"
    total_assigned_seats = 0

    for slice_index in range(slices):
        start_angle = radial_angles[slice_index] - angle_gap / 2
        end_angle = radial_angles[slice_index + 1] + angle_gap / 2

        for piece in range(additional_rows, pieces_per_slice):
            if total_assigned_seats >= int(seats.sum()):
                break

            while remaining_seats == 0 and current_feature < len(sorted_features) - 1:
                current_feature += 1
                remaining_seats = seats[features.index(sorted_features[current_feature])]
                current_color = feature_colors[sorted_features[current_feature]]

            if remaining_seats == 0:
                break

            inner_radius = start_radius + (piece - additional_rows) * piece_depth
            outer_radius = inner_radius + piece_depth - 0.05
            wedge = patches.Wedge(
                (0, 0),
                outer_radius,
                end_angle,
                start_angle,
                width=piece_depth - 0.1,
                facecolor=current_color,
                edgecolor="none",
                linewidth=0,
                zorder=1,
            )
            ax.add_patch(wedge)

            remaining_seats -= 1
            total_assigned_seats += 1

    extended_radius = radius + 1
    for angle in radial_angles:
        x_end = extended_radius * np.cos(np.radians(angle))
        y_end = extended_radius * np.sin(np.radians(angle))
        ax.plot([0, x_end], [0, y_end], color="white", linewidth=3, zorder=3, clip_on=False)

    for piece in range(additional_rows, pieces_per_slice + 1):
        r = start_radius + (piece - additional_rows) * piece_depth - 0.05
        theta = np.linspace(180 + angle_gap / 2, 0 - angle_gap / 2, 300)
        x = r * np.cos(np.radians(theta))
        y = r * np.sin(np.radians(theta))
        ax.plot(x, y, color="white", linewidth=3, zorder=2, clip_on=False)

    ax.set_xlim(-extended_radius - 0.5, extended_radius + 0.5)
    ax.set_ylim(0, extended_radius + 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    legend_patches = []
    for feature in sorted_features:
        seat_count = int(seats[features.index(feature)])
        if seat_count <= 0:
            continue
        legend_patches.append(
            plt.Rectangle((0, 0), 1, 1, color=feature_colors[feature], label=f"{feature} ({seat_count} seats)")
        )
    if legend_patches:
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_title(title or f"{total_seats}-Seat Parliamentary Representation", fontsize=14)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_signed_parliament(
    explanation,
    mode="signed",
    slices=50,
    additional_rows=5,
    show=True,
):
    """Plot positive, negative, or combined DhondtXAI seats."""
    if mode not in {"signed", "positive", "negative"}:
        raise ValueError("mode must be signed, positive, or negative.")

    names = explanation.eligible_alliances
    if mode == "positive":
        seats = [explanation.positive_seats.get(name, 0) for name in names]
        colors = _blue_scale(len(names))
        labels = names
        title = "DhondtXAI Positive Evidence Parliament"
    elif mode == "negative":
        seats = [explanation.negative_seats.get(name, 0) for name in names]
        colors = _red_scale(len(names))
        labels = names
        title = "DhondtXAI Negative Evidence Parliament"
    else:
        labels = []
        seats = []
        colors = []
        blue_colors = _blue_scale(len(names))
        red_colors = _red_scale(len(names))
        for index, name in enumerate(names):
            positive = explanation.positive_seats.get(name, 0)
            negative = explanation.negative_seats.get(name, 0)
            if positive > 0:
                labels.append(f"{name} (+)")
                seats.append(positive)
                colors.append(blue_colors[index])
            if negative > 0:
                labels.append(f"{name} (-)")
                seats.append(negative)
                colors.append(red_colors[index])
        title = "DhondtXAI Signed Evidence Parliament"

    total = int(sum(seats))
    if total <= 0:
        total = explanation.seat_count
    return plot_parliament(
        total_seats=total,
        features=labels,
        seats=seats,
        slices=slices,
        additional_rows=additional_rows,
        colors=colors,
        title=title,
        show=show,
    )


def _default_colors(count):
    colors = [
        "tab:red",
        "tab:blue",
        "tab:green",
        "tab:orange",
        "tab:purple",
        "tab:cyan",
        "tab:pink",
        "tab:olive",
        "gold",
        "teal",
    ]
    if count > len(colors):
        import matplotlib.colors as mcolors

        extra = list(mcolors.CSS4_COLORS.values())
        random.Random(42).shuffle(extra)
        colors += extra[: count - len(colors)]
    return colors[:count]


def _blue_scale(count):
    if count <= 0:
        return []
    cmap = plt.get_cmap("Blues")
    return [cmap(0.45 + 0.45 * (index / max(count - 1, 1))) for index in range(count)]


def _red_scale(count):
    if count <= 0:
        return []
    cmap = plt.get_cmap("Reds")
    return [cmap(0.45 + 0.45 * (index / max(count - 1, 1))) for index in range(count)]
