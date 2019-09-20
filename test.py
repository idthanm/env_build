import numpy as np
import matplotlib.pyplot as plt


def test_reference():
    x = np.linspace(0, 22.375, 100)
    y = -2.42636 * x**2+ 0.11043 * x**3
    print(x, y)
    plt.plot(x, y)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()

def test_plot():
    plt.gca().add_patch(plt.Rectangle((-18, -18), 36, 36, facecolor='none'))

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.show()


if __name__ == "__main__":
    test_plot()