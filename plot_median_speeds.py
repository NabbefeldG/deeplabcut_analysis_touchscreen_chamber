from glob import glob
import os
from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from glob import glob
import pickle


if __name__ == "__main__":
    # load temp data
    with open('all_speeds_mean.pkl', 'rb') as file:
        all_speeds = pickle.load(file)
    #

    target_mice = ["278", "373", "PT374", "PT375", "PT416", "PTD445", "PTD450", "PTD498"]
    all_speeds = {key: all_speeds[key] for key in target_mice}

    plt.close("all")
    for i, mouse_id in enumerate(all_speeds.keys()):
        plt.plot(np.repeat(i, len(all_speeds[mouse_id])), all_speeds[mouse_id], "ok")
        plt.draw()
    #
    plt.xticks(range(len(all_speeds.keys())), all_speeds.keys(), rotation=90)

    plt.ylim([0, 0.001])
    plt.show()
#
