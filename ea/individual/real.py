
import numpy as np
import torch


class Chromosome:
    """
        List of Q-bits.
        Represents list of parameters.
        """

    def __init__(self, chromosome_len):
        self.chromosome_len = chromosome_len
        self.values = np.random.uniform(-1, 1, self.chromosome_len)

    def mutation(self, coef=0.1):
        probs = np.random.normal(0., 1., size=self.chromosome_len)
        self.values = self.values + (coef * probs)


class Individual:
    def __init__(self, model):
        """
        Args:
        chromosome: Chromosome
        """
        self.model = model
        w, b = model.get_weights()
        self.weight_chrom = Chromosome(w.shape[0] * w.shape[1])
        self.bias_chrom = Chromosome(b.shape[0])
        self.chromosomes = [self.weight_chrom, self.bias_chrom]
        self.fitness = float('inf')

    def calc_fitness(self):
        w, b = self.model.get_weights()
        self.model.set_weights(
            self.weight_chrom.values.reshape(w.shape[0], w.shape[1]),
            self.bias_chrom.values
        )

        loss = self.model.evaluate()
        self.fitness = loss
        return loss

    def proceed(self):
        self.calc_fitness()

    def mutation(self, coef):
        for chrom in self.chromosomes:
            chrom.mutation(coef)
        self.proceed()

    def __lt__(self, other):
        return self.fitness < other.fitness
