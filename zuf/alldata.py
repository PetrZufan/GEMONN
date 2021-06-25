
import pickle
from pathlib import Path


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


if __name__ == "__main__":
    import torch
    from zuf.autoencoder import AutoEncoder
    from zuf.qiea.population import Population
    from zuf.main import load_data

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
