from Individual import Individual
import random
from copy import deepcopy


def tournament_selection(population, tau=2):
    """
    Perform tournament selection to choose parents.
    Randomly select tau individuals and choose the best one.

    :param population: List of Individuals.
    :param tau: Number of individuals to select.
    :return: Selected Individual.
    """
    participants = random.sample(population, tau)
    winner = max(participants, key=lambda ind: ind.fitness)
    return deepcopy(winner)


def EA_algorithm(generations,population_size,num_clients,mutation_probability,crossover_probability):
    """
    Perform the Evolutionary Algorithm (EA) to optimize the selection of clients.
    The EA consists of the following steps:
    1. Initialization: Create a population of individuals.
    2. Evaluation: Compute the fitness of each individual.
    3. Selection: Choose parents based on their fitness.
    4. Offspring to create the new population (generational model).
    6. Repeat from step 2 maximum iterations.

    :param generations: Number of generations to run the algorithm.
    :param population_size: Number of individuals in the population.
    :param num_clients: clients selected by each individual.
    :param mutation_probability: Probability of mutation for each individual.
    :param crossover_probability: Probability of crossover for each individual.

    :return global_model: The global model obtained after the EA.
    :return global_accuracy: The validation accuracy of the global model at each generation.
    :return global_loss: The validation loss of the global model at each generation.
    return training_loss: The training loss of the global model at each generation.
    return training_accuracy: The training accuracy of the global model at each generation.
    """

    # Initialize the population
    population = [Individual(genome=random.sample(range(100), k=num_clients)) for _ in range(population_size)]

    for i in range(generations):
        # Evaluate the fitness of each individual
        for individual in population:
            individual.set_fitness(random.random()) #To be done through FL training

        #Here we should average all the models to obtain the global model...

        # Then evaluate the validation accuracy of the global model


        print(f"Generation {i+1}")

        offspring = []
        #Offspring-> offspring size is the same as population size
        for i in range(population_size):
            # Crossover
            if random.random() < crossover_probability:
                parent1 = tournament_selection(population)
                parent2 = tournament_selection(population)
                offspring.append(Individual.crossover(parent1, parent2))
            else:
                parent = tournament_selection(population)
                offspring.append(parent.point_mutation())

        # Replace the population with the new offspring
        population = offspring 

            

            
        
        
