import random

class GeneticAlgorithm:
    def __init__(self) -> None:
        print("GA Activated!")

    def generate_individual(gene_pool):
        # Generate a random permutation of the gene pool
        individual = gene_pool.copy()
        random.shuffle(individual)
        return individual

    def selection(population, fitness_scores):
        # Select parents using tournament selection
        num_parents = len(population)
        parents = []
        for i in range(num_parents):
            competitors = random.sample(list(enumerate(population)), k=2)
            winner = max(competitors, key=lambda x: fitness_scores[x[0]])
            parents.append(winner[1])
        return parents

    def crossover(parent1, parent2):
        # Crossover by selecting a random section of the parents and swapping the genes
        start = random.randint(0, len(parent1) - 1)
        end = random.randint(start + 1, len(parent1))
        child = parent1[start:end] + [g for g in parent2 if g not in parent1[start:end]]
        return child

    def mutate(individual, gene_pool, mutation_rate):
        # Mutate by randomly swapping two genes with the given mutation rate
        mutated_individual = individual.copy()
        for i in range(len(mutated_individual)):
            if random.random() < mutation_rate:
                j = random.randint(0, len(mutated_individual) - 1)
                mutated_individual[i], mutated_individual[j] = mutated_individual[j], mutated_individual[i]
        return mutated_individual
    
    def reproduction(parents, population_size):
        # Reproduce by randomly combining pairs of parents and performing crossover
        offspring = []
        while len(offspring) < population_size:
            father = random.choice(parents)
            mother = random.choice(parents)
            if father != mother:
                child = crossover(father, mother)
                offspring.append(child)
        return offspring

    
    def genetic_algorithm(population_size, num_generations, fitness_fn, gene_pool, mutation_rate=0.01):
        # Initialize the population
        population = [generate_individual(gene_pool) for i in range(population_size)]
        
        # for i in range(num_generations):
        #     # Evaluate the fitness of each individual
        #     fitness_scores = [fitness_fn(individual) for individual in population]
            
        #     # Select the parents for the next generation
        #     parents = selection(population, fitness_scores)
            
        #     # Reproduce to create the next generation
        #     offspring = reproduction(parents, population_size)
            
        #     # Mutate the offspring
        #     mutated_offspring = [mutate(individual, gene_pool, mutation_rate) for individual in offspring]
            
        #     # Replace the old population with the new generation
        #     population = mutated_offspring
        
        # Return the fittest individual
        return max(population, key=fitness_fn)





gene_pool = [1, 2, 3, 4, 5, 6, 7, 8]

def fitness_fn(individual):
    # This fitness function calculates the number of correctly positioned elements in the permutation
    return sum([1 for i, j in zip(individual, gene_pool) if i == j])

geneObj = GeneticAlgorithm()
fittest_permutation = geneObj.genetic_algorithm(population_size = 50, num_generations = 100, fitness_fn = fitness_fn, gene_pool = gene_pool)
print(fittest_permutation)