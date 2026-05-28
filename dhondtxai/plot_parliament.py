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
    snap_seats=True,
    seat_step="auto",
    max_legend_items=12,
    empty_message="No eligible D'Hondt seats",
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
    requested_total_seats = int(total_seats)
    if total_seats < int(seats.sum()):
        total_seats = int(seats.sum())
    if total_seats <= 0:
        raise ValueError("total_seats must be positive.")
    display_total_seats = (
        _recommended_seat_count(int(total_seats), seat_step)
        if snap_seats
        else int(total_seats)
    )
    positive_groups = int(np.count_nonzero(seats))
    display_total_seats = max(display_total_seats, positive_groups)
    display_seats = (
        _rescale_seats(seats, display_total_seats)
        if int(seats.sum()) > 0 and display_total_seats != int(seats.sum())
        else seats.copy()
    )

    if colors is None:
        colors = _default_colors(len(features))
    else:
        colors = list(colors)
        if len(colors) != len(features):
            raise ValueError("Number of colors must match number of features.")

    feature_colors = {feature: colors[i % len(colors)] for i, feature in enumerate(features)}
    order = np.argsort(-display_seats)
    sorted_features = [features[index] for index in order if int(display_seats[index]) > 0]

    pieces_per_slice_without_additional = max(1, math.ceil(display_total_seats / slices))
    pieces_per_slice = pieces_per_slice_without_additional + additional_rows
    angle_gap = 0.2
    radial_angles = np.linspace(180, 0, slices + 1)

    fig, ax = plt.subplots(figsize=(14, 8))
    radius = 10
    piece_depth = radius / pieces_per_slice
    start_radius = piece_depth * additional_rows

    current_feature = 0
    remaining_seats = int(display_seats[features.index(sorted_features[current_feature])]) if sorted_features else 0
    current_color = feature_colors[sorted_features[current_feature]] if sorted_features else "tab:gray"
    total_assigned_seats = 0

    if not sorted_features or int(display_seats.sum()) <= 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis("off")
        ax.text(0.5, 0.5, empty_message, transform=ax.transAxes, ha="center", va="center", fontsize=13)
        ax.set_title(title or f"{requested_total_seats}-Seat Parliamentary Representation", fontsize=14)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    for slice_index in range(slices):
        start_angle = radial_angles[slice_index] - angle_gap / 2
        end_angle = radial_angles[slice_index + 1] + angle_gap / 2

        for piece in range(additional_rows, pieces_per_slice):
            if total_assigned_seats >= int(display_seats.sum()):
                break

            while remaining_seats == 0 and current_feature < len(sorted_features) - 1:
                current_feature += 1
                remaining_seats = int(display_seats[features.index(sorted_features[current_feature])])
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
    for feature in sorted_features[:max_legend_items]:
        seat_count = int(display_seats[features.index(feature)])
        if seat_count <= 0:
            continue
        legend_patches.append(
            plt.Rectangle((0, 0), 1, 1, color=feature_colors[feature], label=f"{feature} ({seat_count} seats)")
        )
    if len(sorted_features) > max_legend_items:
        hidden_count = len(sorted_features) - max_legend_items
        legend_patches.append(
            plt.Rectangle((0, 0), 1, 1, color="lightgray", label=f"+ {hidden_count} smaller groups")
        )
    if legend_patches:
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc="upper left")

    title_text = title or f"{requested_total_seats}-Seat Parliamentary Representation"
    if display_total_seats != requested_total_seats:
        title_text = f"{title_text}\nvisualized as {display_total_seats} seats for readability"
    ax.set_title(title_text, fontsize=14)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_signed_parliament(
    explanation,
    mode="signed",
    slices=50,
    additional_rows=5,
    seat_count=None,
    snap_seats=True,
    seat_step="auto",
    max_legend_items=12,
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
    if seat_count is not None:
        total = int(seat_count)
    return plot_parliament(
        total_seats=total,
        features=labels,
        seats=seats,
        slices=slices,
        additional_rows=additional_rows,
        colors=colors,
        title=title,
        snap_seats=snap_seats,
        seat_step=seat_step,
        max_legend_items=max_legend_items,
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


def _recommended_seat_count(total_seats, seat_step="auto"):
    total_seats = int(total_seats)
    if total_seats <= 0:
        return total_seats
    if seat_step == "auto":
        if total_seats <= 150:
            step = 10
        elif total_seats <= 400:
            step = 50
        else:
            step = 100
    else:
        step = int(seat_step)
        if step <= 0:
            raise ValueError("seat_step must be positive or 'auto'.")
    return max(step, int(round(total_seats / step) * step))


def _rescale_seats(seats, target_total):
    seats = np.asarray(seats, dtype=int)
    target_total = int(target_total)
    if target_total <= 0 or int(seats.sum()) <= 0:
        return np.zeros_like(seats)

    raw = seats.astype(float) / float(seats.sum()) * target_total
    scaled = np.floor(raw).astype(int)
    positive = seats > 0
    scaled[(scaled == 0) & positive] = 1

    while int(scaled.sum()) > target_total:
        candidates = np.where(scaled > 1)[0]
        if len(candidates) == 0:
            break
        index = candidates[np.argmin(raw[candidates] - np.floor(raw[candidates]))]
        scaled[index] -= 1

    remainder = target_total - int(scaled.sum())
    if remainder > 0:
        fractional = raw - np.floor(raw)
        order = np.argsort(-fractional)
        cursor = 0
        while remainder > 0:
            scaled[order[cursor % len(order)]] += 1
            remainder -= 1
            cursor += 1
    return scaled
