import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ea.population import *
from ea.qiea import *

from autoencoder import *


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

    population = Population(pop_size=10)
    qiea = QIEA(population)
    best = qiea.run(10)
    print(best)
    exit()

    hidden_size = 20

    data_size, train_loader, test_loader = load_data()
    model = AutoEncoder(data_size, hidden_size, data_size).cuda()
    print(model)
    train(model, train_loader, data_size)
