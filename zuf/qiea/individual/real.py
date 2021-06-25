
import numpy as np
import torch


class QbitChromosome:
    """
        List of Q-bits.
        Represents list of parameters.
        """

    def __init__(self, chromosome_len):
        self.chromosome_len = chromosome_len
        self.alphas = np.random.uniform(-1, 1, self.chromosome_len)
        self.betas = np.sqrt(np.abs(1 - (self.alphas ** 2))) * np.random.choice([1, -1], size=self.chromosome_len)
        self.collapsed = self.collapse()

    def collapse(self):
        picks = np.random.uniform(0, 1, self.chromosome_len)
        self.collapsed = np.array([alpha if pick else beta for pick, alpha, beta in zip(picks, self.alphas, self.betas)])
        return self.collapsed

    def update(self):
        self.collapse()

    def rotation(self, best_chromosome, coefficient):
        xi_best = np.arctan([beta/alpha for alpha, beta in zip(best_chromosome.alphas, best_chromosome.betas)])
        xi = np.arctan([beta/alpha for alpha, beta in zip(self.alphas, self.betas)])

        xi_0 = np.random.choice([1, -1], size=self.chromosome_len)
        xi_tt = np.where(xi_best >= xi, 1, -1)
        xi_tf = np.sign(self.alphas, best_chromosome.alphas)
        xi_ft = -1 * np.sign(self.alphas, best_chromosome.alphas)
        xi_ff = np.where(xi_best >= xi, 1, -1)

        f = np.array(
            [ x_0 if (x * x_b == 0) or (x * x_b == np.pi/2) or (x * x_b == -1*np.pi/2) else (
                x_tt if (x_b > 0) and (x > 0) else (
                x_tf if (x_b > 0) and (x <= 0) else (
                x_ft if (x_b <= 0) and (x > 0) else
                x_ff
            )))
                for x, x_b, x_0, x_tt, x_tf, x_ft, x_ff
                in zip(xi, xi_best, xi_0, xi_tt, xi_tf, xi_ft, xi_ff)
            ]
        )

        thetas = f * coefficient

        rots = np.array([([np.cos(theta), -1*np.sin(theta)], [np.sin(theta), np.cos(theta)]) for theta in thetas])
        vects = np.column_stack((self.alphas, self.betas))

        vects = np.array([np.dot(rot, vect) for rot, vect in zip(rots, vects)])
        self.alphas = vects[:, 0]
        self.betas = vects[:, 1]

    def mutation(self, probability=0.2):
        probs = np.random.uniform(0, 1, size=self.chromosome_len)
        picks = np.where(probs <= probability, 1, 0)

        new_alphas = np.array([beta if pick else alpha for pick, alpha, beta in zip(picks, self.alphas, self.betas)])
        self.betas = np.array([alpha if pick else beta for pick, alpha, beta in zip(picks, self.alphas, self.betas)])
        self.alphas = new_alphas


class Individual:
    def __init__(self, model):
        """
        Args:
        chromosome: QbitChromosome
        """
        self.model = model
        w, b = model.get_weights()
        self.weight_chrom = QbitChromosome(w.shape[0] * w.shape[1])
        self.bias_chrom = QbitChromosome(b.shape[0])
        self.chromosomes = [self.weight_chrom, self.bias_chrom]
        self.fitness = float('inf')

    def collapse(self):
        return [chrom.collapse() for chrom in self.chromosomes]

    def calc_fitness(self):
        w, b = self.model.get_weights()
        self.model.set_weights(
            self.weight_chrom.collapsed.reshape(w.shape[0], w.shape[1]),
            self.bias_chrom.collapsed
        )

        loss = self.model.evaluate()
        self.fitness = loss
        return loss

    def proceed(self):
        self.collapse()
        self.calc_fitness()

    def rotation(self, best_individual, generation):
        coefficient = np.pi/(100 + generation % 100)
        for i, _ in enumerate(self.chromosomes):
            self.chromosomes[i].rotation(best_individual.chromosomes[i], coefficient)
        self.proceed()

    def mutation(self):
        for chrom in self.chromosomes:
            chrom.mutation()
        self.proceed()
