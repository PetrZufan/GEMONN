
from .individual import *


class Population:
    def __init__(self, model, pop_size=200):
        self.pop_size = pop_size
        self.model = model
        self.population = []
        self.generate_pop()

    def generate_pop(self):
        for _ in np.arange(self.pop_size):
            indv = Individual(self.model)
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

    def observe(self):
        for indv in self.population:
            indv.proceed()

    def rotation(self, best_individual):
        for indv in self.population:
            indv.rotation(best_individual)

    def mutation(self, ratio):
        num = int(self.pop_size * ratio)
        indices = np.random.choice(self.pop_size, num)
        for i in indices:
            self.population[i].mutation()

    def set(self, index, value):
        self.population[index] = value
