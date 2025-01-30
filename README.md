# Optimizing Client Selection in Federated Learning with
Evolutionary Algorithms
Project work for the course of advanced machine learning, year 2024/2025.   

Authors: 
- Dadone Luca
- Fumero Stefano
- Mirenda Andrea  



The focus of the project is Federated Learning with many experiments, run both on the centralized and federated setting, for the CIFAR-100 and Shakespeare datasets.  
PERSONAL CONTRIBUTION !!!!!!!


The repo is organized in 2 main folders:  cifar100 and shakespeare.  
 in cifar-100 and shakespeare, there is the code related to the experiments on the two datasets.  
in cifar 100 in data, you can find the data loader

In particular, all the functions used have been modularized following a similar logic for both the datasets. For the federated setting, a Client and a Server class have been created to manage client update, client selection and sharding. In the utilities, you can find other useful functions to manage checkpointing (to restart executions on Colab when stopped), for plotting results and saving data. In particular, for cifar100, for instance, in federated_utils there is a function that stores models, train and validation accuracies and train and validation losses. This allowed us not to repeat the training when further considerations were needed.  


In each of the two folders containing the code for the two datasets you can find two jupyter notebooks with all the sequence of experiments (one for the centralized setting, the other for the federated one). They can be reproduced and make use of the functions and classes described before. As for readability, we organize them in cycles to group based on the focus of the experiment. For example, when evaluating different values of gamma for the skewed client participation, we report one single cycle for hyperparameter tuniing and training for each gamma for readability.   


For what concerns hyperparameters tuning, different values for lr and wd (and also different schedulers for the centralized version) have been experienced. More details about the result are in Report.md, which reports the results of the experiments for each dataset in the corresponding folder.  

Moreover, in plots_federated and plots_centralized you can find also accuracies and losses for the best set of parameters during tuning. We have not reported them in the report for brevity. 
