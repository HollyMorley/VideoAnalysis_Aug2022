from curlyBrace import curlyBrace
import numpy as np
import matplotlib.colors as mcolors

def add_vertical_brace_curly(ax, y0, y1, x, xoffset, label=None, k_r=0.1, int_line_num=2, fontdict=None, rot_label=0, **kwargs):
    """
    Add a vertical curly brace using the curlyBrace package.
    The brace is drawn at the given x coordinate.
    """
    fig = ax.figure

    fontdict = fontdict or {}
    if 'fontsize' in kwargs:
        fontdict['fontsize'] = kwargs.pop('fontsize')

    p1 = [x, y0]
    p2 = [x, y1]
    # Do not pass the label here.12
    brace = curlyBrace(fig, ax, p1, p2, k_r=k_r, bool_auto=True, str_text=label,
                       int_line_num=int_line_num, fontdict=fontdict or {}, clip_on=False, color='black', **kwargs)

def add_horizontal_brace_curly(ax, x0, x1, y, label=None, k_r=0.1, int_line_num=2, fontdict=None, **kwargs):
    """
    Add a horizontal curly brace using the curlyBrace package.
    The brace is drawn at the given y coordinate.
    """
    fig = ax.figure

    fontdict = fontdict or {}
    if 'fontsize' in kwargs:
        fontdict['fontsize'] = kwargs.pop('fontsize')

    # Swap p1 and p2 so that the brace opens toward the plot.
    p1 = [x1, y]
    p2 = [x0, y]
    brace = curlyBrace(fig, ax, p2, p1, k_r=k_r, bool_auto=True, str_text=label,
                       int_line_num=int_line_num, fontdict=fontdict or {}, clip_on=False, color='black', **kwargs)


def add_cluster_brackets_heatmap(manual_clusters, feature_names, ax, horizontal=True, vertical=True,
                                 base_line_num = 2, label_offset=4, fs=6, distance_from_plot=-0.5):
    #### Add cluster boundaries ####
    cluster_names = {v: k for k, v in manual_clusters['cluster_values'].items()}

    # Compute cluster boundaries based on sorted order.
    cluster_boundaries = {}
    for idx, feat in enumerate(feature_names):
        cl = manual_clusters['cluster_mapping'].get(feat, -1)
        if cl not in cluster_boundaries:
            cluster_boundaries[cl] = {"start": idx, "end": idx}
        else:
            cluster_boundaries[cl]["end"] = idx
    # For each cluster, adjust boundaries by 0.5 (to align with cell edges).
    for i, (cl, bounds) in enumerate(cluster_boundaries.items()):
        # Define boundaries in data coordinates.
        x0, x1 = bounds["start"], bounds["end"]
        y0, y1 = bounds["start"], bounds["end"]

        k_r = 0.1
        span = abs(y1 - y0)
        desired_depth = 0.1  # or any value that gives you the uniform look you want
        k_r_adjusted = desired_depth / span if span != 0 else k_r

        # Alternate the int_line_num value for every other cluster:
        int_line_num = base_line_num + label_offset if i % 2 else base_line_num

        if vertical:
            # Add a vertical curly brace along the left side.
            add_vertical_brace_curly(ax, y0, y1, x=distance_from_plot, xoffset=1, label=cluster_names.get(cl, f"Cluster {cl}"),
                                        k_r=k_r_adjusted, int_line_num=int_line_num, fontsize=fs)
        if horizontal:
            # Add a horizontal curly brace along the top.
            add_horizontal_brace_curly(ax, x0, x1, y=distance_from_plot, label=cluster_names.get(cl, f"Cluster {cl}"),
                                          k_r=k_r_adjusted * -1, int_line_num=int_line_num, fontsize=fs)

# Helper function to darken a hex color
def darken_color(hex_color, factor=0.7):
    # factor < 1 will darken the color
    rgb = mcolors.to_rgb(hex_color)
    dark_rgb = tuple([x * factor for x in rgb])
    return mcolors.to_hex(dark_rgb)

# def get_colors(type):
#     if type == ['APA2','Wash2']:
#         color_1 = "#B11C73" #2FCAD0"
#         color_2 = "#589061"  # "#218EDC" #2F7AD0"
#         colors = (color_1, color_2)
#     elif type == ['APA1','APA2']:
#         color_1 = "#91e3e6"
#         color_2 = "#B11C73"
#         colors = (color_1, color_2)
#
#     return colors

def get_color_phase(phase):
    if phase == 'APA1':
        color = "#D87799"
    elif phase == 'APA2':
        color = "#B11C73" #2FCAD0"
    elif phase == 'Wash1':
        color = "#95CCD8"
    elif phase == 'Wash2':
        color = "#3E9BDD"  # "#218EDC" #2F7AD0"
    else:
        raise ValueError(f"Unknown phase: {phase}")
    return color

def get_color_speedpair(speed):
    if speed == 'LowHigh':
        color = "#288733"
    elif speed == 'LowMid':
        color = "#95BD53"
    elif speed == 'HighLow':
        color = "#C44094" #2FCAD0"
    else:
        raise ValueError(f"Unknown speed: {speed}")
    return color

def get_ls_stride(s):
    if s == -1:
        ls = "-"
    elif s == -2:
        ls = "--"
    elif s == -3:
        ls = ":"
    elif s == 0:
        ls = "-"
    return ls

def get_line_style_mice(m):
    if m == '1035243':
        ls = 'solid'
    elif m == '1035244':
        ls = 'dotted'
    elif m == '1035245':
        ls = 'dashed'
    elif m == '1035246':
        ls = 'dashdot'
    elif m == '1035250':
        ls = (5, (10, 3)) # long dash with offset
    elif m == '1035297':
        ls = (0, (5, 10)) # loosely dashed
    elif m == '1035299':
        ls = (0, (3, 1, 1, 1, 1, 1)) # densely dashdotdotted
    elif m == '1035301':
        ls = (0, (1, 1)) # densely dotted
    else:
        raise ValueError(f"Unknown mouse ID: {m}")
    return ls

def get_marker_style_mice(m):
    if m == '1035243':
        marker = 'o'
    elif m == '1035244':
        marker = '^'
    elif m == '1035245':
        marker = 's'
    elif m == '1035246':
        marker = 'D'
    elif m == '1035250':
        marker = '<'
    elif m == '1035297':
        marker = 'P'
    elif m == '1035299':
        marker = '>'
    elif m == '1035301':
        marker = 'v'
    else:
        raise ValueError(f"Unknown mouse ID: {m}")
    return marker

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))


def create_custom_colormap(lower_color, middle_color, upper_color, scaling=1.0, N=256):
    """
    Create a custom colormap using three specified colors, with a scaling parameter controlling
    the rate at which the colors change from the middle color to the extremes.

    Parameters:
    - lower_color: list or array-like, RGB values (normalized to 0-1) for the lower extreme.
    - middle_color: list or array-like, RGB values (normalized to 0-1) for the center.
    - upper_color: list or array-like, RGB values (normalized to 0-1) for the upper extreme.
    - scaling: float, controls the transition speed. Values > 1 produce a faster change from
               the middle to the extreme, while values < 1 yield a slower transition.
    - N: int, number of discrete color levels (default is 256).

    Returns:
    - cmap: a matplotlib ListedColormap instance.
    """
    colors = []

    for i in range(N):
        t = i / (N - 1)

        if t < 0.5:
            # For lower segment: calculate distance from the middle.
            u = (0.5 - t) / 0.5  # u=1 at t=0, u=0 at t=0.5.
            factor = 1 - (1 - u) ** scaling
            # Blend from middle_color (at u=0) down to lower_color (at u=1).
            color = np.array(middle_color) + factor * (np.array(lower_color) - np.array(middle_color))
        else:
            # For upper segment: calculate distance from the middle.
            u = (t - 0.5) / 0.5  # u=0 at t=0.5, u=1 at t=1.
            factor = 1 - (1 - u) ** scaling
            # Blend from middle_color (at u=0) up to upper_color (at u=1).
            color = np.array(middle_color) + factor * (np.array(upper_color) - np.array(middle_color))

        colors.append(color)

    return mcolors.ListedColormap(colors)


