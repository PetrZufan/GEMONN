
from .individual import *


class Population:
    def __init__(self, pop_size=200, chromosome_len=2):
        self.pop_size = pop_size
        self.chromosome_len = chromosome_len
        self.population = []
        self.generate_pop()

    def generate_pop(self):
        for _ in np.arange(self.pop_size):
            chromosome = QbitChromosome(self.chromosome_len)
            indv = Individual(chromosome)
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
