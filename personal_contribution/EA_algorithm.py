import random
from copy import deepcopy

import torch
import torch.nn as nn

from Individual import Individual
from models.model import LeNet5
from Server import Server
from utils.utils import evaluate


#constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CRITERION = nn.NLLLoss()
MOMENTUM = 0.9 
BATCHSIZE = 64 

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


def client_size(individual, client_sizes):
    """
    Computes the number of total samples for individual
    """
    val = 0
    for client in individual.genome:
        val += client_sizes[client]
    return val


def EA_algorithm(generations,population_size,num_clients,num_classes,crossover_probability, dataset, valid_loader,lr,wd):
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
    :param num_classes: Number of classes for each client (iid or non-iid).
    :param crossover_probability: Probability of crossover for each individual.
    :param dataset: The dataset to be used for training.
    :param valid_loader: The validation loader to evaluate the model.
    :param lr: The learning rate to be used for training.
    :param wd: The weight decay to be used for training.


    :return global_model: The global model obtained after the EA.
    :return val_accuracies: The validation accuracy of the global model at each generation.
    :return val_losses: The validation loss of the global model at each generation.
    :return training_accuracies: The training loss of the global model at each generation.
    :return training_losses: The training accuracy of the global model at each generation.
    :return client_selection_count: The number of times each client was selected in the population.
    """

    # Initialize the population
    population = [Individual(genome=random.sample(range(100), k=num_clients)) for _ in range(population_size)]
    model = LeNet5()
    
    # Create the Server instance:
    server = Server(model,DEVICE)

    shards = server.sharding(dataset.dataset, 100, num_classes)
    client_sizes = [len(shard) for shard in shards]
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    client_selection_count = [0]*100


    for i in range(generations):
       
        # For each of them apply the fed_avg algorithm:
        param_list = []
        averages_acc = []
        average_loss = []
        for individual in population:
            #Update the client selection count
            for client in individual.genome:
                client_selection_count[client] += 1
            
            resulting_model, acc_res, loss_res = server.train_federated(CRITERION, lr, MOMENTUM, BATCHSIZE, wd, individual, shards)
            param_list.append(resulting_model)
            averages_acc.append(acc_res)
            average_loss.append(loss_res)
        

        #Here we should average all the models to obtain the global model...
        averaged_model,  train_loss, train_accuracy = server.fedavg_aggregate(param_list, [client_size(i, client_sizes) for i in population], average_loss, averages_acc)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Update the model with the result of the average:
        model.load_state_dict(averaged_model)


        # Then evaluate the validation accuracy of the global model
        acc, loss = evaluate(model, valid_loader, CRITERION)

        val_accuracies.append(acc)
        val_losses.append(loss)

        print(f"Generation {i+1}, accuracy {acc}, loss {loss}")

        offspring = []
        #Offspring-> offspring size is the same as population size
        for i in range(population_size):
            # Crossover
            if random.random() < crossover_probability:
                parent1 = tournament_selection(population)
                parent2 = tournament_selection(population)
                offspring.append(Individual.crossover(parent1, parent2))
            else:
                #Mutation
                parent = tournament_selection(population)
                parent.point_mutation()
                offspring.append(parent)

        # Replace the population with the new offspring
        population = offspring 
            
    return model, val_accuracies, val_losses,train_accuracies, train_losses, client_selection_count
