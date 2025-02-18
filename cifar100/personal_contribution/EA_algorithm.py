import random
from copy import deepcopy
import os
import sys

import torch
import torch.nn as nn

from Individual import Individual
from models.model import LeNet5
from Server import Server
sys.path.append('../')
from utils.utils import evaluate
from utils.checkpointing_utils import save_checkpoint, load_checkpoint,delete_existing_checkpoints

#constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CRITERION = nn.NLLLoss()
MOMENTUM = 0
BATCHSIZE = 64 
TOTAL_CLIENTS = 100
CHECKPOINTING_PATH = '../checkpoints/'

def tournament_selection_weakest(population, tau=2, p_diver=0.05):
    """
    Perform tournament selection to choose parents.
    Randomly select tau individuals and choose the weakest one.
    Fitness hole to introduce a 5% probability of choosing the fittest individual.


    :param population: List of Individuals.
    :param tau: Number of individuals to select.
    :param p_diver: Probability of choosing the worst individual in the tournament, done for the fitness hole.
    :return: Selected Individual.
    """
    participants = random.sample(population, tau)
    if random.random() < p_diver:
        winner = max(participants, key=lambda ind: ind.fitness)
    else:
      winner = min(participants, key=lambda ind: ind.fitness)
    return deepcopy(winner)


def tournament_selection_fittest(population, tau=2, p_diver=0.05):
    """
    Perform tournament selection to choose parents.
    Randomly select tau individuals and choose the best one.
    Fitness hole to introduce a 5% probability of choosing the weakest individual.


    :param population: List of Individuals.
    :param tau: Number of individuals to select.
    :param p_diver: Probability of choosing the worst individual in the tournament, done for the fitness hole.
    :return: Selected Individual.
    """
    participants = random.sample(population, tau)
    if random.random() < p_diver:
        winner = min(participants, key=lambda ind: ind.fitness)
    else:
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

    #Check if the checkpointing directory exists
    os.makedirs(CHECKPOINTING_PATH, exist_ok=True)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    client_selection_count = [0]*100
    best_model_state = None
    best_val_acc = 0
    last_bests = []
    

    # Initialize the population
    # Shuffle clients before assigning them
    all_clients = list(range(100))
    random.shuffle(all_clients)

    #Disjoint subsets of clients selected by each individual
    population = [
        Individual(genome=all_clients[i * num_clients:(i + 1) * num_clients])
        for i in range(population_size)
    ]
    model = LeNet5()

    #load checkpoint if it exists
    checkpoint_start_step, data_to_load = load_checkpoint(model=model,optimizer=None,hyperparameters=f"LR{lr}_WD{wd}",subfolder="personal_contribution")
    if data_to_load is not None:
        val_accuracies = data_to_load['val_accuracies']
        val_losses = data_to_load['val_losses']
        train_accuracies = data_to_load['train_accuracies']
        train_losses = data_to_load['train_losses']
        client_selection_count = data_to_load['client_selection_count']
        population = [Individual.from_dict(ind) for ind in data_to_load['population']]
    # Create the Server instance:
    server = Server(model,DEVICE)

    shards = server.sharding(dataset.dataset, TOTAL_CLIENTS, num_classes)
    client_sizes = [len(shard) for shard in shards]

    for gen in range(checkpoint_start_step,generations):
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
        acc, loss = evaluate(model, valid_loader)
        if acc > best_val_acc:
                best_val_acc = acc
                best_model_state = deepcopy(model.state_dict())

        val_accuracies.append(acc)
        val_losses.append(loss)

        offspring = []
        #Offspring-> offspring size is the same as population size
        elite = sorted(population, key=lambda ind: ind.fitness, reverse=True)[0]
        offspring.append(elite) #Keep the best individual
        for j in range(population_size-1):
            # Crossover
            if random.random() < crossover_probability:
                parent1 = tournament_selection_fittest(population)
                parent2 = tournament_selection_fittest(population)
                offspring.append(Individual.crossover(parent1, parent2))
            else:
                #Mutation
                parent = tournament_selection_weakest(population)
                parent.mutation()
                offspring.append(parent)

        # Replace the population with the new offspring
        population = offspring 

        #Checkpointing every 10 generations
        if((gen+1)%10==0):
            last_bests.append(best_val_acc)
            if((gen+1)>=30):
                last_bests = last_bests[-3:]
                if(last_bests[0]==last_bests[1] and last_bests[1]==last_bests[2]):
                    #reintroduce diversity
                    random.shuffle(all_clients)
                    population = [
                        Individual(genome=all_clients[i * num_clients:(i + 1) * num_clients])
                        for i in range(population_size-1)
                    ]
                    population.append(elite) #keep the elite
            print(f"Generation {gen+1}, accuracy {best_val_acc}, loss {loss}")
            checkpoint_data = {
                'val_accuracies': val_accuracies,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'train_losses': train_losses,
                'client_selection_count': client_selection_count,
                'population': [individual.to_dict() for individual in population]
            }
            save_checkpoint(model, None, gen+1, f"LR{lr}_WD{wd}", subfolder="personal_contribution", checkpoint_data=checkpoint_data)

    model.load_state_dict(best_model_state) 
    #delete the existing checkpoints for the next execution
    delete_existing_checkpoints(subfolder="personal_contribution")  
    return model, val_accuracies, val_losses,train_accuracies, train_losses, client_selection_count
