# Report results of experiments on the CIFAR-100 dataset
The first part of the experiments focuses on the centralized setting trained and tested on the CIFAR-100 dataset, which contains 60 thousands 32x32 colored images. 
The model architecture used for the experiments is a modified version of LeNet5, taken from [Hsu et al., Federated Visual Classification with Real-World Data Distribution, ECCV 2020] and consists of a CNN similar to LeNet5 which has two 5×5, 64-channel convolution layers, each precedes a 2×2
max-pooling layer, followed by two fully-connected layers with 384 and 192
channels respectively and finally a softmax linear classifier.   

In the first experiments we have done, we noticed that, during training, the model was overfitting because of the visible gap between training and validation accuracy (the same happened for the losses). Therefore, we decided to introduce data augmentation techniques as done in [ Reddi et al., Adaptive federated optimization. ICLR 2021​]. They are very simple and consists of a random crop of the training images of dimension 32x32 with a padding equal to 4 and a random horizontal flip. The techniques applied solved the problem with overfitting.  

For what concerns hyperparameters tuning, we tried different values of lr and wd, together with 3 different schedulers as required by the track. The values for lr and wd have been picked based on a log-uniform distribution over an interval, more details are in the dedicated section in train_centralized.ipynb.  

The hyperparameters values/schedulers over which we tested our model to find the best configuration are:  
- lr: [0.001,0.01,0.1]
- wd: [0.0001,0.001,0.01,0.1]
- schedulers: StepLR (with gamma = 0.1), Cosine Annealing and Exponential LR (with gamma=0.9). 

The same grid of values is used for hyperparameter tuning in all the experiments of the federated setting as well and won't be fully reported again. For what concerns the federated setting, we were not required to use a scheduler.  

For the centralized version on CIFAR-100, we obtained the best accuracy with lr = 0.01, wd = 0.001 and cosine annealing as scheduler. We report here training and validation accuracy for this specific setting after 200 epochs:  

![alt text](images_report/image.png)

We reached, with this configuration, an accuracy of 55.92%, but with this high number of epochs the model clearly started to overfit.

Here we show also train and validation losses:  

![alt text](images_report/image-1.png)  

# Federated CIFAR-100

(All the results of hyperparameter tuning are not report for brevity, but they are all available in the folder federated_plots)

For the federated setting, the tests for hyperparamter tuning brought us to this configuration: lr=0.1 and wd = 0.001, reaching a final accuracy of 47.81%. Here we report training and validation accuracies over 2000 rounds:  

![alt text](images_report/image-2.png)  

While here are the losses:    

![alt text](images_report/image-3.png)

# Evaluate the effect of client participation
We implemented a skewed client sampling: each client has a different probability of being selected at each round, and can be used to simulate settings in which some clients are more “active” than others. Client selection values are sampled according to a Dirichlet distribution parameterized by an hyperparameter ɣ.
Let's test what happens with different values of gamma:  


**gamma = 0.05** <-- Represents extreme heterogeneity. A small number of clients will dominate the selection process, being chosen almost exclusively, while most clients will rarely participate.  


**gamma = 0.5** <-- Introduces moderate heterogeneity. Some clients have higher selection probabilities than others, but the imbalance is not extreme.  


**gamma = 1**   <-- A standard choice for the Dirichlet distribution. This provides a relatively balanced selection with mild heterogeneity.  


**gamma = 5**   <-- Simulates near-uniform participation, where all clients have almost equal probabilities of being selected.  

# Gamma = 5    

Accuracy reached on the test set after 2000 communication rounds: 45.71 %

With gamma = 5, the best hyperparameters found are: lr = 0.1 and wd = 0.0001. We found the exact same results for all the other values of gamma, so we won't report them again. The only exception is for gamma = 0.05, where the best wd was equal to 0.001.  

We report here validation and training accuracies for gamma = 5, followed by validation and training loss. Finally, a bar plot reporting the frequency of client selection is also reported. The same order will be followed also for other values of gamma, so it won't be specified again. Comments will follow at the end.  


# Gamma = 1    

Accuracy reached on the test set after 2000 communication rounds: 45.83 %


 


# Gamma = 0.5  

Accuracy reached on the test set after 2000 communication rounds: 29.02% % 




# Gamma = 0.05  

Accuracy reached on the test set after 2000 communication rounds: 47.04 %  




