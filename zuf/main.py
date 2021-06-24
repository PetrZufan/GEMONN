import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ea.population import *
from ea.qiea import *

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

    hidden_size = 10
    data_size, train_loader, test_loader = load_data(batch_size=64)
    model = AutoEncoder(data_size, hidden_size, data_size).to(device)
    model.set_data(train_loader, data_size)

    population = Population(pop_size=10, model=model)
    qiea = QIEA(population)
    best, all_data = qiea.run(10)
    all_data.to_file("./alldata/alldata.pkl")
    print(best)
