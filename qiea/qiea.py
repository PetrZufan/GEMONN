
from qiea.population import *
from alldata import AllData
import copy


class QIEA:
    def __init__(self, population, children_size):
        self.population = population
        self.pop_size = population.pop_size
        self.children_size = children_size
        self.all_data = AllData()

#    def __init__(self, pop_size, chromosome_len):
#        self.population = Population(pop_size, chromosome_len).generate_pop()
#        self.pop_size = pop_size

    def run(self, generation_num=1, all_data_file="alldata.pkl"):
        # Keep the best individual
        best_individual, best_fitness, _ = self.population.find_best()
        self.all_data.put(copy.deepcopy(self.population))

        for generation in np.arange(generation_num):
            # Print some statistics
            #if i % 10 == 0:
            print('Best Fitness:', round(best_fitness, 5))

            # Quantum Rotation Gate
            self.rotation(best_individual, generation)

            # Mutation
            self.mutation()

            new_individual, new_fitness, _ = self.population.find_best()
            if new_fitness > best_fitness:
                _, _, worst_index = self.population.find_worst()
                self.population.set(worst_index, best_individual)
            else:
                best_fitness = new_fitness
                best_individual = new_individual

            self.all_data.put(copy.deepcopy(self.population))
            # if generation % 5 == 0:
            #    self.all_data.to_file(all_data_file)
        return best_individual, self.all_data

    def rotation(self, best_individual, generation):
        self.population.rotation(best_individual, generation)

    def mutation(self, ratio=0.05):
        self.population.mutation(ratio)
