from glob import glob
from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def load_dlc_csv_results(fPath):
    table = pd.read_csv(fPath, header=0)

    # marker = table.loc[0]
    all_marker_str = ['left_ear', 'right_ear', 'tail',
                      'box_ul', 'box_ur', 'box_lr', 'box_ll']
    d = dict()
    for marker_str in all_marker_str:
        marker_vec = [marker_str in key for key in table.loc[0]]
        for xy_str in ['x', 'y']:
            xy_vec = [j == xy_str for j in list(table.loc[1])]
            d[marker_str+'_'+xy_str] = np.array(table.iloc[2:, np.array(marker_vec) & np.array(xy_vec)].apply(pd.to_numeric)).flatten()
        #
    #

    return pd.DataFrame(d)
#


# cmaps = {}
#
# gradient = np.linspace(0, 1, 256)
# gradient = np.vstack((gradient, gradient))
#
# def plot_color_gradients(category, cmap_list):
#     # Create figure and adjust figure height to number of colormaps
#     nrows = len(cmap_list)
#     figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
#     fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
#     fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
#                         left=0.2, right=0.99)
#     axs[0].set_title(f'{category} colormaps', fontsize=14)
#
#     for ax, name in zip(axs, cmap_list):
#         ax.imshow(gradient, aspect='auto', cmap=mpl.colormaps[name])
#         ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
#                 transform=ax.transAxes)
#
#     # Turn off *all* ticks & spines, not just the ones with colormaps.
#     for ax in axs:
#         ax.set_axis_off()
#
#     # Save colormap list for later.
#     cmaps[category] = cmap_list
# #


if __name__ == "__main__":
    folder = r"C:\Users\nabbefeld\Documents\GitHub\deeplabcut_analysis_touchscreen_chamber\data"
    # file = r"278_WIN_20220524_090701DLC_resnet50_testOct30shuffle1_10000_filtered.csv"
    # file = r"373_WIN_20220524_093853DLC_resnet50_testOct30shuffle1_10000_filtered.csv"
    # file = r"373_WIN_20220524_093853DLC_resnet50_testOct30shuffle1_10000.csv"
    file = r"278_WIN_20220524_090701DLC_resnet50_testOct30shuffle1_10000.csv"
    df = load_dlc_csv_results(path.join(folder, file))


    """
    This is something I saw and we need to be aware of when interpreting the data!!!
    Some sessions are recorded at 15Hz and Some at 30Hz !
    I dont know if that's just the difference between the two setups or not
    But this is crucial when interpreting the speed of mice!    
    """

    print(df)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(df.left_ear_x, df.left_ear_y)
    ax[0, 0].invert_yaxis()
    ax[0, 0].title.set_text('Left_ear')

    ax[0, 1].plot(df.right_ear_x, df.right_ear_y)
    ax[0, 1].invert_yaxis()
    ax[0, 1].title.set_text('Right_ear')

    ax[1, 0].plot(df.tail_x, df.tail_y)
    ax[1, 0].invert_yaxis()
    ax[1, 0].title.set_text('Tail')

    mouse_center = np.array([(df.left_ear_x + df.right_ear_x + df.tail_x) / 3,
                             (df.left_ear_y + df.right_ear_y + df.tail_y) / 3])

    ax[1, 1].plot(mouse_center[0, :], mouse_center[1, :])
    ax[1, 1].invert_yaxis()
    ax[1, 1].title.set_text('mouse_center')

    bin_f = 40
    img = np.zeros(np.int32([np.ceil(1920 / bin_f), np.ceil(1080 / bin_f)]))

    img_binned = np.int32(np.round(mouse_center / bin_f))

    for i in range(mouse_center.shape[1]):
        img[img_binned[0, i], img_binned[1, i]] += 1
    #

    img = img / np.sum(img[:])

    plt.figure()
    plt.imshow(img.T, vmin=0, vmax=np.percentile(img[:], 99), cmap=mpl.colormaps['Reds'])
    plt.colorbar()

    box = np.array([[np.nanmedian(df.box_ul_x),
                     np.nanmedian(df.box_ur_x),
                     np.nanmedian(df.box_lr_x),
                     np.nanmedian(df.box_ll_x),
                     np.nanmedian(df.box_ul_x)],
                    [np.nanmedian(df.box_ul_y),
                     np.nanmedian(df.box_ur_y),
                     np.nanmedian(df.box_lr_y),
                     np.nanmedian(df.box_ll_y),
                     np.nanmedian(df.box_ul_y)]
                    ]) / bin_f

    plt.plot(box[0, :], box[1, :], 'k')

    plt.savefig(path.splitext(file)[0]+'_heatmap.png')

    plt.show()
#
