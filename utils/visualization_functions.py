import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, MaxNLocator
import statsmodels.formula.api as smf
import os
import seaborn as sns
from matplotlib.lines import Line2D


plt.style.context('seaborn-talk')
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['pdf.fonttype']=42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11

def get_n_colors(cmap, n):
    if n == 1:
        return mpl.colors.rgb2hex(cmap(0.5), keep_alpha=True)
    else:
        return [mpl.colors.rgb2hex(cmap(i), keep_alpha=True) for i in np.linspace(0.25, 0.75, n)]

from matplotlib.colors import LinearSegmentedColormap

cud_skyblue = "#56B4E9"
cmap_skyblue = LinearSegmentedColormap.from_list("cud_skyblue_white", ["white", cud_skyblue])
cud_orange = "#E69F00"
cmap_orange = LinearSegmentedColormap.from_list("cud_orange_white", ["white", cud_orange])

# for two groups
young_shade = get_n_colors(cmap_skyblue, 1)
old_shade = get_n_colors(cmap_orange, 1)
palette2 = [young_shade, old_shade]

palette2_swarm = [cud_skyblue, cud_orange]

# for 10 animals (with teddy)
young_shades = get_n_colors(cmap_skyblue, 5)
old_shades = get_n_colors(cmap_orange, 5)
palette10 = list(np.concatenate([np.asarray(young_shades), np.asarray(old_shades)]))

palette10_swarm = np.concatenate([[cud_skyblue]*5, [cud_orange]*5])

subj_order = ['pippin', 'chip', 'bobrick', 'timothy', 'teddy', 'archibald', 'hugo', 'herman', 'reginald', 'tony']
subj_abbr_labels = ['P', 'C', 'B', 'Ti', 'Te', 'A', 'Hu', 'He', 'R', 'To']

# without teddy:
# young_shades = get_n_colors(cmap_skyblue, 4)
# old_shades = get_n_colors(cmap_orange, 5)
# palette10 = list(np.concatenate([np.asarray(young_shades), np.asarray(old_shades)]))

# palette10_swarm = np.concatenate([[cud_skyblue]*4, [cud_orange]*5])


# subj_order = ['pippin', 'chip', 'bobrick', 'timothy', 'archibald', 'hugo', 'herman', 'reginald', 'tony']
# subj_abbr_labels = ['P', 'C', 'B', 'Ti', 'A', 'Hu', 'He', 'R', 'To']



# create a function to visualize the decode for a particular epoch given relevant info and time stamps
def plot_decode_2d(df, subj, place_bin_size, interior_place_bin_centers, time_mask, title, time_units='s', position_type='line'):
    # df needs to have the position information and the MAP information

    fig, ax = plt.subplots(figsize=(4, 3.5))

    if subj in ['pippin', 'archibald', 'hugo', 'herman', 'chip']:
        uni = 'ucsf'
    if subj in ['bobrick', 'reginald', 'timothy', 'tony', 'teddy']:
        uni = 'uw'

    xs = interior_place_bin_centers[:, 0]
    ys = interior_place_bin_centers[:, 1]
    widths = np.ones(len(interior_place_bin_centers))*place_bin_size[0]
    heights = np.ones(len(interior_place_bin_centers))*place_bin_size[1]        

    base_patches = [
        Rectangle((x - w/2, y - h/2), w, h)
        for x, y, w, h in zip(xs, ys, widths, heights)
    ]

    base_collection = PatchCollection(base_patches, color='lightgray', ec='lightgray')
    ax.add_collection(base_collection)

    # plot the MAP values on top of this in a colormap gradient that's based on time
    map_centers = df.loc[time_mask, ['map_x', 'map_y']].values

    if uni == 'ucsf':
        xs = map_centers[:, 1]
        ys = -map_centers[:, 0]
        widths = np.ones(len(map_centers))*place_bin_size[1]
        heights = np.ones(len(map_centers))*place_bin_size[0]
    if uni == 'uw':
        xs = map_centers[:, 0]
        ys = map_centers[:, 1]
        widths = np.ones(len(map_centers))*place_bin_size[0]
        heights = np.ones(len(map_centers))*place_bin_size[1]       

    map_patches = [
        Rectangle((x - w/2, y - h/2), w, h)
        for x, y, w, h in zip(xs, ys, widths, heights)
    ]

    color_values = np.linspace(0, 1, len(map_patches))

    map_collection = PatchCollection(map_patches, cmap='viridis', alpha=1)
    map_collection.set_array(color_values)
    ax.add_collection(map_collection)

    ax.autoscale()

    # plot actual position on top of this
    actual_positions = df.loc[time_mask, ['pos_x', 'pos_y']].values

    if position_type == 'line':
        if uni == 'ucsf':
            plt.plot(actual_positions[:, 1], -actual_positions[:, 0], color='red', lw=2)
        if uni == 'uw':
            plt.plot(actual_positions[:, 0], actual_positions[:, 1], color='red', lw=2)
    if position_type == 'scatter':
        if uni == 'ucsf':
            plt.scatter(actual_positions[:, 1], -actual_positions[:, 0], color='red')
        if uni == 'uw':
            plt.scatter(actual_positions[:, 0], actual_positions[:, 1], color='red')

    plt.title(title)

    # --- scale bar lengths (DATA UNITS) ---
    x_len = 10    # x-axis scale length
    y_len = 10     # y-axis scale length

    # --- placement as fraction of axis ---
    x_frac = 0.05
    y_frac = 0.05

    # convert axis fraction → data coordinates
    x0 = ax.get_xlim()[0] + x_frac * (ax.get_xlim()[1] - ax.get_xlim()[0])
    y0 = ax.get_ylim()[0] + y_frac * (ax.get_ylim()[1] - ax.get_ylim()[0])

    # horizontal (x) scale bar
    plt.plot([x0, x0 + x_len], [y0, y0],
            color='black', lw=2)

    # vertical (y) scale bar
    plt.plot([x0, x0], [y0, y0 + y_len],
            color='black', lw=2)

    # labels
    plt.text(x0 + x_len / 2, y0 - 2,
            f'{x_len} cm',
            ha='center', va='top')

    plt.text(x0 - 2, y0 + y_len / 2,
            f'{y_len} cm',
            ha='right', va='center')

    plt.axis('off')

    cbar = fig.colorbar(map_collection, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'time ({time_units})')
    cbar.set_ticks([0, 1])
    if time_units == 's':
        cbar.set_ticklabels([0, len(map_centers)*4 / 1000])
    if time_units == 'ms':
        cbar.set_ticklabels([0, len(map_centers)*4])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(size=0)


def auto_set_yticks(ax):
    # autocalculate ylims and yticks, ensure that the top tick aligns with the top of the axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', integer=False, prune=None))

    plt.draw()  # force tick calculation

    # Step 2: snap axis limits to include the top tick
    ticks = ax.get_yticks()
    ymin, ymax = ax.get_ylim()

    if ticks[-1] < ymax:        # if the last tick is below the top
        ticks = list(ticks) + [ymax]  # add the top limit as a tick

    ax.set_yticks(ticks)         # update major ticks
    ax.set_ylim(ymin, ticks[-1]) # ensure axis limit matches top tick

def custom_set_ylim(ax, ymin, ymax):
    # autocalculate yticks based on custom ylim, ensure that the top tick aligns with the top of the axis
    ax.set_yticks([])
    ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', integer=False, prune=None))
    ax.set_ylim([ymin, ymax])

def custom_set_xlim(ax, xmin, xmax):
    # autocalculate xticks based on custom ylim, ensure that the top tick aligns with the top of the axis
    ax.set_xticks([])
    ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', integer=False, prune=None))
    ax.set_xlim([xmin, xmax])


def plot_all_subjs_young_aged(data, y_col, y_label, title, save_dir, fig_name, plot_kind='bar', set_custom_ylim=False, ymin=None, ymax=None, showfliers=True):
    
    if plot_kind == 'bar':
        g = sns.catplot(data=data, x='subj', order=subj_order, y=y_col,
                hue='age', hue_order=['young', 'aged'], palette=palette2_swarm, kind='bar', errorbar='se', height=2.5, aspect=1)
    if plot_kind == 'violin':
        g = sns.catplot(data=data, x='subj', order=subj_order, y=y_col,
                hue='age', hue_order=['young', 'aged'], palette=palette2_swarm, kind='violin', errorbar='se', height=2.5, aspect=1, inner=None, linewidth=0.5)
    if plot_kind == 'box':
        g = sns.catplot(data=data, x='subj', order=subj_order, y=y_col,
                hue='age', hue_order=['young', 'aged'], palette=palette2_swarm, kind='box', height=2.5, aspect=1, showfliers=showfliers)

    ax = g.axes[0][0]

    if plot_kind == 'violin':
        # Compute medians
        medians = data.groupby("subj", sort=False)[y_col].median().reindex(subj_order).values

        # Overlay median bars
        for i, median in enumerate(medians):
            ax.hlines(median, i - 0.05, i + 0.05, color="black", linewidth=2)  # horizontal bar

    # plt.xticks(ticks=plt.xticks()[0], labels=subj_order, rotation=45, ha='right', va='top', rotation_mode='anchor')
    ax.set_xticks(ticks=ax.get_xticks(), labels=subj_abbr_labels)

    ax.set_ylabel(y_label)
    ax.set_xlabel('Subject')
    ax.set_title(title)
    plt.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    g._legend.remove()
    ax.legend(handles, ["Young", "Aged"], title="Age")

    try:
        # calculate LME stats
        processed_data = data[['age', 'subj', y_col]].dropna()
        md = smf.mixedlm(f"{y_col} ~ age", processed_data, groups="subj")
        mdf = md.fit()
        ax.text(0.5, -0.4, f'age effect pval: {mdf.pvalues["age[T.young]"]}', ha='center', va='bottom', fontsize=8, transform=ax.transAxes)
    except Exception as e:
        print(e)

    if set_custom_ylim:
        custom_set_ylim(g.ax, ymin, ymax)
    else:
        auto_set_yticks(g.ax)

    if save_dir is not None:
        if '.png' in fig_name:
            plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fig_name))

    # plt.show()

    return g


def add_figure_scalebar(
    fig, ref_ax,
    x_len, y_len,
    x_label=None, y_label=None,
    origin=(0.05, -0.06),   # outside bottom-left
    lw=2,
    color='black',
    label_pad=0.4          # fraction of bar thickness
):
    """
    Draw figure-level scale bars using ref_ax as the data ruler.
    """

    # ---- data → pixel conversion ----
    p0 = ref_ax.transData.transform((0, 0))
    px = ref_ax.transData.transform((x_len, 0))
    py = ref_ax.transData.transform((0, y_len))

    dx_pix = px[0] - p0[0]
    dy_pix = py[1] - p0[1]

    fig_w, fig_h = fig.get_size_inches() * fig.dpi
    dx = dx_pix / fig_w
    dy = dy_pix / fig_h

    x0, y0 = origin

    # ---- scale bars ----
    fig.add_artist(Line2D(
        [x0, x0 + dx], [y0, y0],
        transform=fig.transFigure,
        lw=lw, color=color
    ))

    fig.add_artist(Line2D(
        [x0, x0], [y0, y0 + dy],
        transform=fig.transFigure,
        lw=lw, color=color
    ))

    # ---- automatic label spacing ----
    pad_x = label_pad * dy
    pad_y = label_pad * dx

    if x_label is None:
        x_label = f'{x_len}'
    if y_label is None:
        y_label = f'{y_len}'

    fig.text(
        x0 + dx / 2, y0 - pad_x,
        x_label,
        ha='center', va='top'
    )

    fig.text(
        x0 - pad_y, y0 + dy / 2,
        y_label,
        ha='right', va='center'
    )