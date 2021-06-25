
import pickle
from pathlib import Path

import numpy as np


class AllData:
    def __init__(self):
        self.populations = []

    def put(self, population):
        self.populations.append(population)

    def get(self):
        return self.populations

    def to_file(self, filename):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'wb') as file:
            pickle.dump(self.populations, file, pickle.HIGHEST_PROTOCOL)

    def from_file(self, filename):
        with open(filename, 'rb') as file:
            self.populations = pickle.load(file)
        return self

    def fitness_to_file(self, filename):
        bests = [population.find_best() for population in self.populations]
        bests = np.array([[i, value] for i, (_, value, _) in enumerate(bests)])
        x = bests[:, 0]
        y = bests[:, 1]

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as file:
            for i, fit in bests:
                file.write(str(i)+','+str(fit)+'\n')


if __name__ == "__main__":
    import torch
    from autoencoder import AutoEncoder
    from qiea.population import Population
    from main import load_data

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_size, train_loader, test_loader = load_data(batch_size=64)
    model = AutoEncoder(data_size, 2, data_size).to(device)
    model.set_data(train_loader, data_size)
    population1 = Population(model, 3)
    population2 = Population(model, 3)
    population3 = Population(model, 3)

    alldata = AllData()
    alldata.put(population1)
    alldata.put(population2)
    alldata.put(population3)

    filename = "./alldata/alldata.pkl"
    alldata.to_file(filename)

    loaded = AllData()
    loaded.from_file(filename)
