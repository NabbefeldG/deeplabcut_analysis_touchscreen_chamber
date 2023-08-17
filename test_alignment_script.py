import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data = np.array([[1, 4, 2, 1, 1],
                     [4, 1, 1, 2, 4]])

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot([-5, 5], [0, 0], "--k")
    ax[0].plot([0, 0], [-5, 5], "--k")

    ax[0].plot(data[0, :], data[1, :])
    ax[0].set_ylim([-5, 5])
    ax[0].set_xlim([-5, 5])
    ax[0].set_title("Original Coordinates")


    data2 = data.copy()
    data2[0, :] -= data2[0, 0]
    data2[1, :] -= data2[1, 0]

    ax[1].plot([-5, 5], [0, 0], "--k")
    ax[1].plot([0, 0], [-5, 5], "--k")
    ax[1].plot(data2[0, :], data2[1, :])
    ax[1].set_ylim([-5, 5])
    ax[1].set_xlim([-5, 5])
    ax[1].set_title("Translated")


    data3 = data2.copy()


    alpha = np.arctan(data2[1, 1] / data2[0, 1])
    R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

    # data3 = np.dot(data3.T, R)
    data3 = np.dot(R.T, data3)
    # data3 = np.dot(data3.T, R).T


    ax[2].plot([-5, 5], [0, 0], "--k")
    ax[2].plot([0, 0], [-5, 5], "--k")
    ax[2].plot(data3[0, :], data3[1, :])
    ax[2].set_ylim([-5, 5])
    ax[2].set_xlim([-5, 5])
    ax[2].set_title("Translated + Rotated")

    ax[0].axis("equal")
    ax[1].axis("equal")
    ax[2].axis("equal")
    plt.savefig("Alignment_visualization.png", dpi=500)
    plt.show()
#

