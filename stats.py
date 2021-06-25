
import os
import matplotlib.pyplot as plt
import numpy as np

from alldata import AllData


class Stats:
    def __init__(self, all_data=None):
        self.all_data = all_data

    def conv(self, file, title, ylim=0.3, xlim=100):
        bests = [population.find_best() for population in self.all_data.populations]
        bests = np.array([[i, value] for i, (_, value, _) in enumerate(bests)])
        x = bests[:, 0]
        y = bests[:, 1]

        plt.clf()
        plt.plot(x, y, linewidth=0.5)
        plt.xlabel('generation')
        plt.ylabel('fitness')
        plt.ylim(0, ylim)
        plt.xlim(0, xlim)
        plt.title(title)
        plt.savefig(file, bbox_inches='tight')

    def conv_all(self, all_data_list, file, title, ylim=0.3, xlim=100):
        plt.clf()
        for all_data in all_data_list:
            bests = [population.find_best() for population in all_data.populations]
            bests = np.array([[i, value] for i, (_, value, _) in enumerate(bests)])
            x = bests[:, 0]
            y = bests[:, 1]
            plt.plot(x, y, linewidth=0.5)
        plt.xlabel('generation')
        plt.ylabel('fitness')
        plt.ylim(0, ylim)
        plt.xlim(0, xlim)
        plt.title(title)
        plt.savefig(file, bbox_inches='tight')


if __name__ == "__main__":
    directory = "../clasic/alldata/classic/"
    all_data_list = [AllData().from_file(directory + file) for file in os.listdir(directory)]
    stats = Stats()
    stats.conv_all(all_data_list, "conv.png", "title")
