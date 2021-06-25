import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from qiea.population import *
from qiea.qiea import *
from qiea.individual.real import Individual as RealIndv
from qiea.individual.binary import Individual as BinIndv

from autoencoder import *


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(batch_size=-1):
    train_data = datasets.MNIST(
        root="data/",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size if batch_size > 0 else train_data.data.shape[0],
        shuffle=True
    )

    test_data = datasets.MNIST(
        root="data/",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size if batch_size > 0 else test_data.data.shape[0],
        shuffle=False
    )

    size = train_data.data.shape[1] * train_data.data.shape[2]

    return size, train_dataloader, test_dataloader


if __name__ == '__main__':

    hidden_layer_size = 10
    population_size = 10
    max_generation = 10
    individual_type = BinIndv

    data_size, train_loader, test_loader = load_data(batch_size=64)
    model = AutoEncoder(data_size, hidden_layer_size, data_size).to(device)
    model.set_data(train_loader, data_size)

    population = Population(pop_size=population_size, model=model, individual_type=individual_type)
    qiea = QIEA(population)
    best, all_data = qiea.run(max_generation)
    all_data.to_file("./alldata/alldata.pkl")
    print(best)
