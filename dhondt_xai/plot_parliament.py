
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import random

def plot_parliament(total_seats, features, seats, slices=50, additional_rows=5):
    # Validate inputs
    if len(features) != len(seats):
        raise ValueError("Number of features and seats must match.")
    
    # Renkleri belirleme
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'lime', 'pink']
    
    # Eğer feature sayısı renk sayısından fazlaysa, renk listesini genişletiyoruz
    if len(features) > len(colors):
        import matplotlib.colors as mcolors
        all_colors = list(mcolors.CSS4_COLORS.values())
        random.shuffle(all_colors)
        colors += all_colors[:len(features) - len(colors)]
    else:
        colors = colors[:len(features)]
    
    # Meclis düzeni parametreleri
    pieces_per_slice_without_additional = math.ceil(total_seats / slices)
    pieces_per_slice = pieces_per_slice_without_additional + additional_rows
    
    # Açı hesaplamaları
    angle_per_slice = 180 / slices
    
    # Figür oluştur ve yarım daire çizin
    fig, ax = plt.subplots(figsize=(14, 8))
    radius = 10
    
    # Yeni yarıçap hesaplamaları
    piece_depth = radius / pieces_per_slice
    start_radius = piece_depth * additional_rows
    
    # Açısal boşluk
    angle_gap = 0.2
    radial_angles = np.linspace(180, 0, slices + 1)
    
    current_feature = 0
    current_color = colors[current_feature]
    remaining_seats = seats[current_feature]
    
    total_assigned_seats = 0
    
    for slice_index in range(slices):
        start_angle = radial_angles[slice_index] - angle_gap / 2
        end_angle = radial_angles[slice_index + 1] + angle_gap / 2

        for piece in range(additional_rows, pieces_per_slice):
            if total_assigned_seats >= total_seats:
                break

            if remaining_seats == 0 and current_feature < len(seats) - 1:
                current_feature += 1
                remaining_seats = seats[current_feature]
                current_color = colors[current_feature]

            inner_radius = start_radius + (piece - additional_rows) * piece_depth
            outer_radius = inner_radius + piece_depth - 0.05
            wedge = patches.Wedge(
                (0, 0), outer_radius, end_angle, start_angle,
                width=piece_depth - 0.1, facecolor=current_color, edgecolor='none', linewidth=0, zorder=1
            )
            ax.add_patch(wedge)

            remaining_seats -= 1
            total_assigned_seats += 1

            if total_assigned_seats >= total_seats:
                break

    extended_radius = radius + 1

    for angle in radial_angles:
        x_end = extended_radius * np.cos(np.radians(angle))
        y_end = extended_radius * np.sin(np.radians(angle))
        ax.plot([0, x_end], [0, y_end], color='white', linewidth=3, linestyle='-', zorder=3, clip_on=False)

    for piece in range(additional_rows, pieces_per_slice + 1):
        r = start_radius + (piece - additional_rows) * piece_depth - 0.05
        theta = np.linspace(180 + angle_gap / 2, 0 - angle_gap / 2, 300)
        x = r * np.cos(np.radians(theta))
        y = r * np.sin(np.radians(theta))
        ax.plot(x, y, color='white', linewidth=3, linestyle='-', zorder=2, clip_on=False)

    ax.set_xlim(-extended_radius - 0.5, extended_radius + 0.5)
    ax.set_ylim(0, extended_radius + 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plt.tight_layout()

    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=colors[i], label=f"{features[i]} ({seats[i]} MV)") for i in range(len(features))]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f"{total_seats} Sandalyeli Meclis Temsili", fontsize=14)
    plt.show()
