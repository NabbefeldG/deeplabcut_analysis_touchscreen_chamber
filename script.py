from glob import glob
import os
from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from glob import glob
import pickle


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


def _align_xy(data_x, data_y, offset, R, s):
    data_x -= offset[0]
    data_y -= offset[1]

    data = np.dot(R, np.array([data_x, data_y])) / s
    # data = np.dot(np.array([data_x, data_y]).T, R.T).T

    return data
#


def align_touchscreen_df(data):
    temp_ul = [np.nanmedian(data.box_ul_x), np.nanmedian(data.box_ul_y)]
    temp_ur = [np.nanmedian(data.box_ur_x), np.nanmedian(data.box_ur_y)]

    dx = (temp_ur[0] - temp_ul[0])
    dy = (temp_ur[1] - temp_ul[1])

    s = np.sqrt(np.power(dx, 2) + np.power(dy, 2))

    alpha = np.arctan(dy / dx)
    R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]]).T

    data_keys = np.array(data.keys())
    data_keys = [t.replace("_x", "") for t in data_keys]
    data_keys = [t.replace("_y", "") for t in data_keys]
    data_keys = np.unique(data_keys)
    for key in data_keys:
        temp = _align_xy(data[key+"_x"], data[key+"_y"], temp_ul, R, s)
        data[key + "_x"] = temp[0, :]
        data[key + "_y"] = temp[1, :]
    #

    return data
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
    # file = r"278_WIN_20220524_090701DLC_resnet50_testOct30shuffle1_10000.csv"
    file = r"278_WIN_20220525_085119DLC_resnet50_2023-08-14_TCPhase2Aug14shuffle1_500000_filtered.csv"
    all_files = glob("data\*.csv")

    all_speeds_median = dict()
    all_speeds_mean = dict()
    for file in all_files[7:]:
        # df = load_dlc_csv_results(path.join(folder, file))
        df = load_dlc_csv_results(file)
        df = align_touchscreen_df(df)

        """
        This is something I saw and we need to be aware of when interpreting the data!!!
        Some sessions are recorded at 15Hz and Some at 30Hz !
        I dont know if that's just the difference between the two setups or not
        But this is crucial when interpreting the speed of mice!    
        """

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

        # mouse_center[mouse_center < 0] = 0
        # mouse_center[mouse_center > 1] = 1

        # bin_f = 40
        bin_f = 0.02
        offset = np.round(1/bin_f)
        offset = np.int32(0)
        # img = np.zeros(np.int32([np.ceil(1920 / bin_f), np.ceil(1080 / bin_f)]))
        img = np.zeros(np.int32([np.ceil(1 / bin_f), np.ceil(1 / bin_f)] + 2*offset))

        img_binned = np.int32(np.round(mouse_center / bin_f)+offset)

        for i in range(mouse_center.shape[1]):
            try:
                img[img_binned[0, i], img_binned[1, i]] += 1
            except:
                pass
            #
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
                        ]) / bin_f - 0.5 + offset

        plt.plot(box[0, :], box[1, :], 'k')
        # plt.savefig(path.splitext(file)[0]+'_heatmap.png')

        plt.xlim([0, 1 / bin_f])
        plt.ylim([0, 1 / bin_f])
        plt.gca().invert_yaxis()

        # plt.savefig(path.splitext(file)[0]+'_heatmap.png')

        os.makedirs("heatmaps", exist_ok=True)
        plt.savefig(path.join("heatmaps", path.splitext(path.basename(file))[0] + '_heatmap.png'))



        # also look at the median speed for mice

        fps = 15  # ToDo: THIS IS JUST A GUESS FOR NOW!!!

        # print(mouse_center)
        mean_speed_xy = np.mean(np.diff(mouse_center, 1), 1)
        median_speed_xy = np.median(np.diff(mouse_center, 1), 1)

        median_speed = np.sqrt(np.square(median_speed_xy[0]) + np.square(median_speed_xy[1])) * fps
        mean_speed = np.sqrt(np.square(mean_speed_xy[0]) + np.square(mean_speed_xy[1])) * fps

        mouse_id = path.basename(file).split("_")[0]
        if mouse_id not in all_speeds_mean.keys():
            all_speeds_mean[mouse_id] = list()
            all_speeds_median[mouse_id] = list()
        #

        all_speeds_mean[mouse_id].append(mean_speed)
        all_speeds_median[mouse_id].append(median_speed)
        print(median_speed)
        # plt.show()
    #

    # Open a file and use dump()
    with open('all_speeds_mean.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(all_speeds_mean, file)
    #

    # Open a file and use dump()
    with open('all_speeds_median.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(all_speeds_median, file)
    #

    print(all_speeds_median)

    plt.close("all")
    for i, mouse_id in enumerate(all_speeds_median.keys()):
        plt.plot(np.repeat(i, len(all_speeds_median[mouse_id])), all_speeds_median[mouse_id], "ok")
        plt.draw()
    #
    plt.show()
#
