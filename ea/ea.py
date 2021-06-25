
from ea.population import *
from alldata import AllData
import copy


class EA:
    def __init__(self, population, children_count):
        self.population = population
        self.pop_size = population.pop_size
        self.all_data = AllData()
        self.children_count = children_count

#    def __init__(self, pop_size, chromosome_len):
#        self.population = Population(pop_size, chromosome_len).generate_pop()
#        self.pop_size = pop_size

    def run(self, generation_num=1, all_data_file="alldata.pkl"):
        # Keep the best individual
        best_individual, best_fitness, _ = self.population.find_best()
        self.all_data.put(copy.deepcopy(self.population))

        for generation in np.arange(generation_num):
            print('Best Fitness:', round(best_fitness, 5))

            # Mutation
            self.mutation()

            new_individual, new_fitness, _ = self.population.find_best()
            if new_fitness > best_fitness:
                pass
            else:
                best_fitness = new_fitness
                best_individual = new_individual

            self.all_data.put(copy.deepcopy(self.population))
            # if generation % 5 == 0:
                # self.all_data.to_file(all_data_file)
        return best_individual, self.all_data

    def mutation(self, coef=1.0):
        self.population.mutation(coef, self.children_count)
