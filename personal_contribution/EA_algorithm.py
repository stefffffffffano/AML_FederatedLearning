import random
from copy import deepcopy
import os
import torch
import torch.nn as nn
from statistics import mean
import numpy as np
from Individual import Individual
from cifar100_loader import CIFAR100DataLoader
from model import LeNet5
from Server import Server


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

def evaluate(model, dataloader, DEVICE, criterion):
    with torch.no_grad():
        model.train(False) # Set Network to evaluation mode
        running_corrects = 0
        losses = []
        for data, targets in dataloader:
            data = data.to(DEVICE)        # Move the data to the GPU
            targets = targets.to(DEVICE)  # Move the targets to the GPU
            # Forward Pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            # Get predictions
            _, preds = torch.max(outputs.data, 1)
            # Update Corrects
            running_corrects += torch.sum(preds == targets.data).data.item()
            # Calculate Accuracy
            accuracy = running_corrects / float(len(dataloader.dataset))

    return accuracy*100, mean(losses)


def EA_algorithm(generations,population_size,num_clients,crossover_probability, dataset, valid_loader):
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
    :param crossover_probability: Probability of crossover for each individual.

    :return global_model: The global model obtained after the EA.
    :return global_accuracy: The validation accuracy of the global model at each generation.
    :return global_loss: The validation loss of the global model at each generation.
    return training_loss: The training loss of the global model at each generation.
    return training_accuracy: The training accuracy of the global model at each generation.
    """

    # Initialize the population
    population = [Individual(genome=random.sample(range(100), k=num_clients)) for _ in range(population_size)]
    model = LeNet5()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.NLLLoss()
    # Create the Server instance:
    server = Server(model,DEVICE,"server_data")
    num_classes = 100
    shards = server.sharding(dataset.dataset, num_clients, num_classes)
    client_sizes = [len(shard) for shard in shards]
    LR = 0.01
    batchsize = 64
    WD = 0.001
    MOMENTUM = 0.9


    for i in range(generations):
        # Select randomly 3 individuals:
        selected_individuals = population
        # For each of them apply the fed_avg algorithm:
        
        param_list = []
        averages_acc = []
        average_loss = []
        for choosen_individual in selected_individuals:
            resulting_model, acc_res, loss_res = server.train_federated(criterion, LR, MOMENTUM, batchsize, WD, choosen_individual, shards)
            param_list.append(resulting_model)
            averages_acc.append(acc_res)
            average_loss.append(loss_res)
        

        #Here we should average all the models to obtain the global model...
        averaged_model,  global_avg_loss, global_avg_accuracy = server.fedavg_aggregate(param_list, [client_size(i, client_sizes) for i in selected_individuals], average_loss, averages_acc)
        # Update the model with the result of the average:
        model.load_state_dict(averaged_model)


        # Then evaluate the validation accuracy of the global model
        acc, loss = evaluate(model, valid_loader, DEVICE, criterion)

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
                parent = tournament_selection(population)
                #parent.point_mutation()
                offspring.append(parent)

        # Replace the population with the new offspring
        population = offspring 
            

            
if __name__ == '__main__':
    #10% of the dataset kept for validation
    BATCH_SIZE = 64
    data_loader = CIFAR100DataLoader(batch_size=BATCH_SIZE, validation_split=0.1, download=True, num_workers=4, pin_memory=True)
    trainloader, validloader, testloader = data_loader.train_loader, data_loader.val_loader, data_loader.test_loader
    EA_algorithm(10, 3, 100, 0.7, trainloader, validloader)

