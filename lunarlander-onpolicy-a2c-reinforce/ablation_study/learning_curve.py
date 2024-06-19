import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


class LearningCurvePlot:

    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self, y, label=None):
        if label is not None:
            self.ax.plot(y, label=label)
        else:
            self.ax.plot(y)

    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    def add_hline(self, height, label):
        self.ax.axhline(height, ls='--', c='k', label=label)

    def save(self, name='LearningCurveTest.png'):
        self.ax.legend()
        self.fig.savefig(name, dpi=300)


def smooth_curve(y, window, poly=1):
    return savgol_filter(y, window, polyorder=poly)

