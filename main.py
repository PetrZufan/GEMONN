import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from alldata import AllData
from qiea.population import Population as QPopulation
from qiea.qiea import QIEA
from qiea.individual.real import Individual as RealIndv
from qiea.individual.binary import Individual as BinIndv

from ea.population import Population as CPopulation
from ea.ea import EA
from ea.individual.real import Individual as ClassicIndv

from autoencoder import *
from stats import Stats

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
    pid = os.getpid()

    parser = argparse.ArgumentParser(description='QIEA')
    parser.add_argument('--gen', type=int, default=500, help='The maximal iteration of the algorithm')
    parser.add_argument('--pop', type=int, default=4, help='The population size')
    parser.add_argument('--child', type=int, default=8, help='The children count')
    parser.add_argument('--hid', type=int, default=300, help='The number of hidden units of an auto-encoder')
    parser.add_argument('--alg', type=str, default="ea", help='The algorithm used. Values: ea, grad.')
    parser.add_argument('--indv', type=str, default="real", help='Type of individual encoding. Values: real, bin, classic.')
    parser.add_argument('--svpath', type=str, default='./alldata/')
    parser.add_argument('--svfile', type=str, default='alldata_' + str(pid))

    args = parser.parse_args()

    hidden_layer_size = args.hid
    population_size = args.pop
    children_size = args.child
    max_generation = args.gen
    individual_type = \
        BinIndv if (args.indv == 'bin') else (
        RealIndv if (args.indv == 'real') else (
        ClassicIndv
    ))
    file = args.svpath + args.svfile

    data_size, train_loader, test_loader = load_data(batch_size=64)
    model = AutoEncoder(data_size, hidden_layer_size, data_size).to(device)
    model.set_data(train_loader, data_size)

    if args.alg == "grad":
        epochs = max_generation / (data_size/64)
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            model.grad_train()
            model.test()
        print("Done!")


    elif individual_type == ClassicIndv:
        population = CPopulation(pop_size=population_size, model=model, individual_type=individual_type)
        ea = EA(population, children_size)
        best, all_data = ea.run(max_generation, file)
        #all_data.to_file(file)
        all_data.fitness_to_file(file + '.txt')
    else:
        population = QPopulation(pop_size=population_size, model=model, individual_type=individual_type)
        qiea = QIEA(population, children_size)
        best, all_data = qiea.run(max_generation, file)
        #all_data.to_file(file)
        all_data.fitness_to_file(file + '.txt')


    #stats = Stats(all_data)
    #stats.conv(file + '.png', "title", xlim=10)

    # print(best)
