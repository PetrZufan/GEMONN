import numpy as np

from ea.individual.real import *


class Population:
    def __init__(self, model, individual_type, pop_size=200):
        self.pop_size = pop_size
        self.model = model
        self.population = []
        self.generate_pop(individual_type)

    def generate_pop(self, individual_type):
        for _ in np.arange(self.pop_size):
            indv = individual_type(self.model)
            indv.proceed()
            self.population.append(indv)
        return self.population

    def find_best(self):
        fitnesses = np.array([indv.fitness for indv in self.population])
        min_value = min(fitnesses)
        min_index = fitnesses.argmin()
        return self.population[min_index], min_value, min_index

    def find_worst(self):
        fitnesses = np.array([indv.fitness for indv in self.population])
        max_value = max(fitnesses)
        max_index = fitnesses.argmin()
        return self.population[max_index], max_value, max_index

    def select(self, child_count):
        return np.random.choice(self.population, self.pop_size)

    def mutation(self, coef, child_count):
        children = self.select(child_count)
        for child in children:
            child.mutation(coef)
        np.append(children, self.population)
        children.sort()
        self.population = children[:self.pop_size]

    def set(self, index, value):
        self.population[index] = value
