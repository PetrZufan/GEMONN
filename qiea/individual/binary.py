
import numpy as np
import torch


class QbitGene:
    """
    List of Q-Bits.
    Represents one parameter.
    """
    def __init__(self, bit_len=5):
        self.bit_len = bit_len
        self.degrees = np.pi/4 * np.ones(bit_len)
        self.alphas = np.sin(self.degrees)
        self.betas = np.cos(self.degrees)
        self.collapsed, self.value = self.collapse()

    def collapse(self, bound=10):
        pick = np.random.uniform(0, 1, self.bit_len)
        self.collapsed = np.where(pick >= (self.alphas ** 2), 1, 0)
        return self.collapsed, self.to_value(bound)

    def to_value(self, bound=10):
        base = np.array([2 ** l for l in range(self.bit_len)])
        dec_value = np.dot(self.collapsed, base)
        self.value = -bound + dec_value / ((2 ** self.bit_len) - 1) * 2 * bound
        return self.value

    def update(self):
        self.collapse()

    def rotation(self, best_gene, fitness_flag):
        criteria = (self.collapsed + best_gene.collapsed) == 1
        deltas = criteria * (0.01 * np.pi)
        sgns = np.zeros(self.bit_len)

        current_best_bit_flag = self.collapsed - best_gene.collapsed
        fitness_flags = np.ones(self.bit_len) * fitness_flag
        fitness_flags = np.where(fitness_flags > 0, 1, -1)
        alpha_beta_pos = (self.alphas * self.betas) > 0
        alpha_beta_neg = (self.alphas * self.betas) < 0
        alpha_zero = self.alphas == 0
        beta_zero = self.betas == 0
        # if alpha * beta>0
        sgns += current_best_bit_flag * fitness_flags * alpha_beta_pos
        # if alpha * beta<0
        sgns += (-1) * current_best_bit_flag * fitness_flags * alpha_beta_neg
        # if alpha = 0
        # Generate +1 or -1 at random
        direction = np.random.choice([1, -1], size=self.bit_len)
        criteria = current_best_bit_flag * fitness_flags * alpha_zero < 0
        sgns += criteria * direction
        # if beta = 0
        criteria = current_best_bit_flag * fitness_flags * beta_zero > 0
        sgns += criteria * direction
        # Calculate shift angles
        angles = deltas * sgns
        # Calculate new angles
        self.degrees = self.degrees - angles
        self.alphas = np.sin(self.degrees)
        self.betas = np.cos(self.degrees)

    def mutation(self):
        picks = np.random.uniform(0, 1, size=self.bit_len)
        alpha_flags = self.alphas < picks
        beta_flags = self.betas < picks
        self.degrees = self.degrees - alpha_flags * beta_flags * np.pi / 2
        self.alphas = np.sin(self.degrees)
        self.betas = np.cos(self.degrees)


class QbitChromosome:
    """
        List of QbitsGenes.
        Represents list of parameters.
        """
    def __init__(self, chromosome_len, bit_len=5):
        self.chromosome_len = chromosome_len
        self.bit_len = bit_len
        self.genes = np.array([QbitGene(bit_len) for _ in np.arange(chromosome_len)])

        # warning: Values not updated when called genes[0].collapse().
        # In that case, call self.update() first.
        self.collapsed, self.values = self.collapse()

    def collapse(self, bound=10):
        self.collapsed = np.array([gene.collapse()[0] for gene in self.genes])
        self.values = np.array([gene.to_value(bound) for gene in self.genes])
        return self.collapsed, self.values

    def update(self):
        self.collapsed = np.array([gene.collapsed for gene in self.genes])
        self.values = np.array([gene.value for gene in self.genes])
        return self.collapsed, self.values

    def rotation(self, best_chromosome, fitness_flag):
        for i, _ in enumerate(self.genes):
            self.genes[i].rotation(best_chromosome.genes[i], fitness_flag)
        # self.update() # already done by Individual.proceed()

    def mutation(self):
        for gene in self.genes:
            gene.mutation()


class Individual:
    def __init__(self, model):
        """
        Args:
        chromosome: QbitChromosome
        """
        self.model = model
        w, b = model.get_weights()
        self.weight_chrom = QbitChromosome(w.shape[0]*w.shape[1])
        self.bias_chrom = QbitChromosome(b.shape[0])
        self.chromosomes = [self.weight_chrom, self.bias_chrom]
        self.fitness = float('inf')

    def collapse(self, bound=10):
        return [chrom.collapse(bound) for chrom in self.chromosomes]

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
        self.collapse()
        self.calc_fitness()

    def rotation(self, best_individual, generation):
        fitness_flag = self.fitness > best_individual.fitness
        for i, _ in enumerate(self.chromosomes):
            self.chromosomes[i].rotation(best_individual.chromosomes[i], fitness_flag)
        self.proceed()

    def mutation(self):
        for chrom in self.chromosomes:
            chrom.mutation()
        self.proceed()
