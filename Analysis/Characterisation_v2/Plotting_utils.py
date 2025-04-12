from curlyBrace import curlyBrace

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

def get_colors(type):
    if type == ['APA2','Wash2']:
        color_1 = "#2FCAD0"
        color_2 = "#2F7AD0"
        colors = (color_1, color_2)

    return colors