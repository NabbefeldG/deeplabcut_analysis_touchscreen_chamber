import numpy as np
import matplotlib.pyplot as plt


def align_touchscreen_data(data):
    # define ul corner as [0, 0]
    data[0, :] -= data[0, 0]
    data[1, :] -= data[1, 0]

    # Then rotate, so that the Monitor is along the abscissa
    alpha = np.arctan(data[1, 1] / data[0, 1])
    R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]]).T
    data = np.dot(R, data)

    return data
#


if __name__ == "__main__":
    data = np.array([[1, 4, 2, 1, 1],
                     [4, 1, 1, 2, 4]])

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].plot([-5, 5], [0, 0], "--k")
    ax[0].plot([0, 0], [-5, 5], "--k")

    ax[0].plot(data[0, :], data[1, :])
    ax[0].set_ylim([-5, 5])
    ax[0].set_xlim([-5, 5])
    ax[0].set_title("Original Coordinates")

    data = align_touchscreen_data(data)

    ax[1].plot([-5, 5], [0, 0], "--k")
    ax[1].plot([0, 0], [-5, 5], "--k")
    ax[1].plot(data[0, :], data[1, :])
    ax[1].set_ylim([-5, 5])
    ax[1].set_xlim([-5, 5])
    ax[1].set_title("Translated + Rotated")

    ax[0].axis("equal")
    ax[1].axis("equal")
    plt.show()
#

