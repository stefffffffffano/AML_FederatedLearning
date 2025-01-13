import random

#usage: parent1 = Individual(genome=random.sample(range(total_clients), k=num_selected_clients))


class Individual:
    def __init__(self, genome, total_clients=100,number_selected_clients = 10):
        """
        Initialize an Individual.

        :param genome: List of selected clients (subset of integers).
        :param total_clients: Total number of available clients (default 100).
        :param number_selected_clients: Number of clients to be selected (default 10).
        """
        self.genome = genome
        self.fitness = None  # Fitness will be computed separately
        self.total_clients = total_clients
        self.number_selected_clients = number_selected_clients
    
    def set_fitness(self, fitness_value):
        """
        Set the fitness value for the individual.
        
        :param fitness_value: Float value representing the fitness (e.g., loss or accuracy).
        """
        self.fitness = fitness_value

    def mutate(self):
        """
        Mutate the genome by changing 2 to 4 clients randomly.
        Ensures that the selected clients remain disjoint.
        """
        num_changes = random.randint(2, 4)  # Number of mutations
        available_clients = set(range(self.total_clients)) - set(self.genome)  # Clients not in genome

        # Remove random clients from the genome
        to_remove = random.sample(self.genome, k=num_changes)
        for client in to_remove:
            self.genome.remove(client)

        # Add new random clients from the available set
        to_add = random.sample(available_clients, k=num_changes)
        self.genome.extend(to_add)

    @staticmethod
    def crossover(parent1, parent2):
        """
        Perform crossover between two parents.
        Select half genes from parent1 and half from parent2.
        Ensures that the genome is disjoint and valid.

        :param parent1: First parent Individual.
        :param parent2: Second parent Individual.
        :return: New Individual (offspring).
        """
        half_size = len(parent1.genome) // 2

        # Randomly select half genes from each parent
        genome1_part = random.sample(parent1.genome, k=half_size)
        genome2_part = [gene for gene in parent2.genome if gene not in genome1_part]

        # Combine to form new genome
        new_genome = genome1_part + genome2_part[:len(parent1.genome) - half_size]

        #Ensure that the genome is long enough
        if(len(new_genome)< parent1.number_selected_clients):
            new_genome = new_genome + random.sample(range(parent1.total_clients), k=parent1.number_selected_clients - len(new_genome))

        return Individual(genome=new_genome, total_clients=parent1.total_clients)