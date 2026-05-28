import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def _validate_language(language):
    if language != "en":
        raise ValueError("Only English output is supported.")


def plot_parliament(
    total_seats,
    features,
    seats,
    slices=50,
    additional_rows=0,
    colors=None,
    title=None,
    snap_seats=True,
    seat_step="auto",
    max_legend_items=12,
    inner_radius_ratio=0.36,
    seat_label="MPs",
    show_scaled_counts=False,
    empty_message="No eligible D'Hondt seats",
    quality_note=None,
    language="en",
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
    _validate_language(language)
    if np.any(seats < 0):
        raise ValueError("seats must be non-negative.")
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
        colors = _paper_colors(len(features))
    else:
        colors = list(colors)
        if len(colors) != len(features):
            raise ValueError("Number of colors must match number of features.")

    feature_colors = {feature: colors[i % len(colors)] for i, feature in enumerate(features)}
    order = np.argsort(-display_seats)
    sorted_features = [features[index] for index in order if int(display_seats[index]) > 0]

    effective_slices = int(slices) + max(0, int(additional_rows))
    if effective_slices <= 0:
        raise ValueError("slices plus additional_rows must be positive.")
    seat_rows = max(1, math.ceil(display_total_seats / effective_slices))
    angle_gap = 0.15
    radial_angles = np.linspace(180, 0, effective_slices + 1)

    radius = 10
    inner_radius_ratio = float(inner_radius_ratio)
    if not 0 <= inner_radius_ratio < 0.75:
        raise ValueError("inner_radius_ratio must be in [0, 0.75).")
    start_radius = radius * inner_radius_ratio
    piece_depth = (radius - start_radius) / seat_rows

    if not sorted_features or int(display_seats.sum()) <= 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis("off")
        ax.text(0.5, 0.5, empty_message, transform=ax.transAxes, ha="center", va="center", fontsize=13)
        empty_title = title or f"{requested_total_seats}-Seat Parliamentary Representation"
        ax.set_title(empty_title, fontsize=14)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    fig, ax = plt.subplots(figsize=(16, 7.5))
    current_feature = 0
    remaining_seats = int(display_seats[features.index(sorted_features[current_feature])]) if sorted_features else 0
    current_color = feature_colors[sorted_features[current_feature]] if sorted_features else "tab:gray"
    total_assigned_seats = 0

    for slice_index in range(effective_slices):
        start_angle = radial_angles[slice_index] - angle_gap / 2
        end_angle = radial_angles[slice_index + 1] + angle_gap / 2

        for piece in range(seat_rows):
            if total_assigned_seats >= int(display_seats.sum()):
                break

            while remaining_seats == 0 and current_feature < len(sorted_features) - 1:
                current_feature += 1
                remaining_seats = int(display_seats[features.index(sorted_features[current_feature])])
                current_color = feature_colors[sorted_features[current_feature]]

            if remaining_seats == 0:
                break

            inner_radius = start_radius + piece * piece_depth
            outer_radius = inner_radius + piece_depth - 0.05
            wedge = patches.Wedge(
                (0, 0),
                outer_radius,
                end_angle,
                start_angle,
                width=piece_depth - 0.1,
                facecolor=current_color,
                edgecolor="white",
                linewidth=0.35,
                zorder=1,
            )
            ax.add_patch(wedge)

            remaining_seats -= 1
            total_assigned_seats += 1

    extended_radius = radius + 1
    for angle in radial_angles:
        x_start = start_radius * np.cos(np.radians(angle))
        y_start = start_radius * np.sin(np.radians(angle))
        x_end = extended_radius * np.cos(np.radians(angle))
        y_end = extended_radius * np.sin(np.radians(angle))
        ax.plot([x_start, x_end], [y_start, y_end], color="white", linewidth=2.4, zorder=3, clip_on=False)

    for piece in range(seat_rows + 1):
        r = start_radius + piece * piece_depth - 0.05
        theta = np.linspace(180 + angle_gap / 2, 0 - angle_gap / 2, 300)
        x = r * np.cos(np.radians(theta))
        y = r * np.sin(np.radians(theta))
        ax.plot(x, y, color="white", linewidth=2.4, zorder=2, clip_on=False)

    ax.set_xlim(-extended_radius - 0.5, extended_radius + 0.5)
    ax.set_ylim(0, extended_radius + 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    legend_patches = []
    for feature in sorted_features[:max_legend_items]:
        seat_count = int(display_seats[features.index(feature)])
        original_seat_count = int(seats[features.index(feature)])
        if seat_count <= 0:
            continue
        if seat_count != original_seat_count and show_scaled_counts:
            label = f"{feature} ({original_seat_count} {seat_label}, shown as {seat_count})"
        else:
            label = f"{feature} ({seat_count} {seat_label})"
        legend_patches.append(
            plt.Rectangle((0, 0), 1, 1, color=feature_colors[feature], label=label)
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
        suffix = f"visualized as {display_total_seats} seats for readability"
        title_text = f"{title_text}\n{suffix}"
    ax.set_title(title_text, fontsize=14)
    if quality_note:
        ax.text(
            0.01,
            0.98,
            quality_note,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="darkred",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "darkred"},
        )
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_signed_parliament(
    explanation,
    mode="signed",
    slices=50,
    additional_rows=0,
    seat_count=None,
    snap_seats=True,
    seat_step="auto",
    max_legend_items=12,
    palette="paper",
    inner_radius_ratio=0.36,
    seat_label="MPs",
    show_scaled_counts=False,
    show_quality_note=False,
    language="en",
    show=True,
):
    """Plot positive, negative, or combined DhondtXAI seats."""
    if mode not in {"signed", "positive", "negative"}:
        raise ValueError("mode must be signed, positive, or negative.")
    if palette not in {"paper", "signed", "distinct", "positive_negative"}:
        raise ValueError("palette must be 'paper', 'signed', 'distinct', or 'positive_negative'.")
    _validate_language(language)

    names = explanation.eligible_alliances
    if mode == "positive":
        seats = [explanation.positive_seats.get(name, 0) for name in names]
        labels = names
        signs = ["positive"] * len(labels)
        title = "DhondtXAI Positive Evidence Parliament"
    elif mode == "negative":
        seats = [explanation.negative_seats.get(name, 0) for name in names]
        labels = names
        signs = ["negative"] * len(labels)
        title = "DhondtXAI Negative Evidence Parliament"
    else:
        labels = []
        seats = []
        signs = []
        for index, name in enumerate(names):
            positive = explanation.positive_seats.get(name, 0)
            negative = explanation.negative_seats.get(name, 0)
            if positive > 0:
                labels.append(f"{name} (+)")
                seats.append(positive)
                signs.append("positive")
            if negative > 0:
                labels.append(f"{name} (-)")
                seats.append(negative)
                signs.append("negative")
        title = "DhondtXAI Signed Evidence Parliament"

    colors = _palette_colors(labels, signs, palette)
    quality_note = _quality_note(explanation, language) if show_quality_note else None

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
        inner_radius_ratio=inner_radius_ratio,
        seat_label=seat_label,
        show_scaled_counts=show_scaled_counts,
        quality_note=quality_note,
        language=language,
        show=show,
    )


def _default_colors(count):
    # A compact Glasbey-style qualitative palette. The early colors are chosen
    # for high perceptual separation, and later colors are generated by
    # golden-angle hue rotation to stay usable when the number of groups grows.
    colors = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#F0E442",
        "#56B4E9",
        "#E69F00",
        "#332288",
        "#88CCEE",
        "#44AA99",
        "#117733",
        "#999933",
        "#DDCC77",
        "#CC6677",
        "#882255",
        "#AA4499",
        "#661100",
        "#6699CC",
        "#AA4466",
        "#4477AA",
        "#228833",
        "#EE6677",
        "#BBBBBB",
        "#000000",
    ]
    if count > len(colors):
        colors += [_golden_angle_color(index) for index in range(count - len(colors))]
    return colors[:count]


def _paper_colors(count):
    colors = [
        "#ff00ff",
        "#0000ff",
        "#00d7ff",
        "#ff9900",
        "#ffd700",
        "#00cc44",
        "#008000",
        "#ff0000",
        "#7f00ff",
        "#ff66aa",
        "#00a6a6",
        "#8b4513",
        "#4b0082",
        "#ff6f00",
        "#00ff99",
        "#3366ff",
        "#cc0066",
        "#999900",
        "#cc3300",
        "#0099ff",
        "#66cc00",
        "#cc99ff",
        "#ffcc00",
        "#333333",
    ]
    if count > len(colors):
        colors += [_golden_angle_color(index) for index in range(count - len(colors))]
    return colors[:count]


def _palette_colors(labels, signs, palette):
    labels = list(labels)
    signs = list(signs)
    if palette == "paper":
        return _paper_colors(len(labels))
    if palette == "distinct":
        return _default_colors(len(labels))

    positive = _positive_colors(max(1, signs.count("positive")))
    negative = _negative_colors(max(1, signs.count("negative")))
    pos_i = 0
    neg_i = 0
    colors = []
    for sign in signs:
        if sign == "negative":
            colors.append(negative[neg_i % len(negative)])
            neg_i += 1
        else:
            colors.append(positive[pos_i % len(positive)])
            pos_i += 1
    return colors


def _positive_colors(count):
    colors = [
        "#0072B2",
        "#009E73",
        "#56B4E9",
        "#44AA99",
        "#332288",
        "#117733",
        "#6699CC",
        "#4477AA",
        "#228833",
        "#88CCEE",
    ]
    if count > len(colors):
        colors += [_golden_angle_color(index) for index in range(count - len(colors))]
    return colors[:count]


def _negative_colors(count):
    colors = [
        "#D55E00",
        "#CC6677",
        "#882255",
        "#E69F00",
        "#AA4466",
        "#AA4499",
        "#661100",
        "#EE6677",
        "#CC79A7",
        "#999933",
    ]
    if count > len(colors):
        colors += [_golden_angle_color(index + 97) for index in range(count - len(colors))]
    return colors[:count]


def _quality_note(explanation, language):
    _validate_language(language)
    ratio = getattr(explanation, "projection_residual_ratio", 0.0)
    bucket = getattr(explanation, "projection_residual_attribution", 0.0)
    if ratio < 0.10 and abs(bucket) <= 1e-12:
        return None
    if ratio >= 0.50:
        return f"Caution: high projection correction ({ratio:.0%})."
    return f"Projection correction: {ratio:.0%}."


def _golden_angle_color(index):
    import colorsys

    hue = (0.61803398875 * (index + 1)) % 1.0
    saturation = 0.72
    value = 0.86 if index % 2 == 0 else 0.70
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return "#{:02x}{:02x}{:02x}".format(int(red * 255), int(green * 255), int(blue * 255))


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
