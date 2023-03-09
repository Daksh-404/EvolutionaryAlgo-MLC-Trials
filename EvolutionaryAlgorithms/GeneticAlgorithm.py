import random

class GeneticAlgorithm:
    def __init__(self, chromosome_size, conditional_entropy_matrix) -> None:
        print("GA Activated!")
        self.chromosome_size = chromosome_size
        self.conditional_entropy_matrix = conditional_entropy_matrix
        self.gene_pool = [i + 1 for i in range(chromosome_size)]

    def fitness_fn(self, chromosome):
        # This fitness function calculates the upper triangular matrix of Conditional Entropy
        # after it is rearranged according to the given chromosome / permutation

        # Reaarangement
        new_objective_fn = []
        for idx1 in chromosome:
            row_conditional_entropy = []
            for idx2 in chromosome:
                row_conditional_entropy.append(self.conditional_entropy_matrix[idx1 - 1][idx2 - 1])
            new_objective_fn.append(row_conditional_entropy)

        # Sum of upper triangular matrix
        fitness_val = 0
        for i in range(self.chromosome_size):
            for j in range(self.chromosome_size):
                if (i < j):
                    fitness_val = fitness_val + new_objective_fn[i][j]

        return fitness_val
    
    def generate_individual(self):
        # Generate a random permutation of the gene pool
        individual = self.gene_pool.copy()
        random.shuffle(individual)
        return individual

    def tournament_selection(self, population, fitness_scores):
        # Select parents using tournament selection
        num_parents = len(population)
        parents = []
        for i in range(num_parents):
            competitors = random.sample(list(enumerate(population)), k=2)
            print(competitors)
            winner = max(competitors, key=lambda x: fitness_scores[x[0]])
            parents.append(winner[1])
        return parents

    def roulette_selection(self, population, fitness_scores):
        num_parents = len(population)
        parents = []
        for i in range(num_parents):
            prob = random.uniform(0, sum(fitness_scores))
            for j, fitness_val in enumerate(fitness_scores):
                if prob <= 0:
                    break
                prob -= fitness_val
            parents.append(population[j])
            print(j)
        return parents

    def crossover_helper(self, parent1, parent2, start, end):
        child = []
        for i in range(start):
            if parent2[i] not in parent1[start : end]:
                child.append(parent2[i])

        child = child + parent1[start : end]
        for i in range(start, self.chromosome_size):
            if parent2[i] not in parent1[start : end]:
                child.append(parent2[i])
        
        return child
    
    def ordered_crossover(self, population, parent1, parent2, crossover_rate):
        # Ordered Crossover by selecting a random section of the parents
        start = random.randint(0, self.chromosome_size - 1)
        end = random.randint(start + 1, self.chromosome_size)
        child1 = []
        child2 = []
        if random.random() < crossover_rate:
            child1 = self.crossover_helper(parent1, parent2, start, end)
            child2 = self.crossover_helper(parent2, parent1, start, end)
        else:
            child1 = random.sample(population, k = 1)[0]
            child2 = random.sample(population, k = 1)[0]
        return child1, child2
        

    def mutate(self, chromosome, mutation_rate):
        # Mutate by randomly swapping two genes with the given mutation rate
        mutated_individual = chromosome.copy()
        if random.random() < mutation_rate:
            i = random.randint(0, len(mutated_individual) - 1)
            j = random.randint(0, len(mutated_individual) - 1)
            mutated_individual[i], mutated_individual[j] = mutated_individual[j], mutated_individual[i]
        return mutated_individual
    
    def reproduction(self, parents, population, population_size, crossover_rate):
        # Reproduce by randomly combining pairs of parents and performing crossover
        temp_parents = parents.copy()
        offspring = []
        while len(offspring) < population_size:
            father, mother = random.sample(temp_parents, k = 2)
            child1, child2 = self.ordered_crossover(population, father, mother, crossover_rate)
            offspring.append(child1)
            offspring.append(child2)
            temp_parents.remove(father)
            temp_parents.remove(mother)
            print(len(offspring))
        return offspring

    def refine_population(self, offspring, parents, elitism_rate, population_size):
        prev_gen = parents.copy()
        new_gen = offspring.copy()
        prev_gen.sort(key = self.fitness_fn, reverse = True)
        new_gen.sort(key = self.fitness_fn, reverse = True)

        new_population = prev_gen[:elitism_rate] + new_gen[:population_size - elitism_rate]
        return new_population
    
    def genetic_algorithm(self, population_size = 50, num_generations = 100, mutation_rate = 0.01, crossover_rate = 0.9, elitism_rate = 5):
        # Initialize the population
        population = [self.generate_individual() for i in range(population_size)]
        
        for i in range(num_generations):
            # Evaluate the fitness of each individual
            fitness_scores = [self.fitness_fn(chromosome) for chromosome in population]
            # Select the parents for the next generation
            parents = self.roulette_selection(population, fitness_scores)
            
            # Reproduce to create the next generation
            offspring = self.reproduction(parents, population, population_size, crossover_rate)
            
            # Mutate the offspring
            mutated_offspring = [self.mutate(chromosome, mutation_rate) for chromosome in offspring]
            
            # Generate the new population using elitism
            population = self.refine_population(mutated_offspring, parents, elitism_rate, population_size)
            
        
        # Return the fittest individual
        return max(population, key = self.fitness_fn)


