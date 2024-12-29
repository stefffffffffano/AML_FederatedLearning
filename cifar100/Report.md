# Report for the analysis performed on the CIFAR-100 dataset
**Centralized baseline**  

For what concerns the centralized baseline, we first looked for the best hyperparameters for training (the ones able to obtain an higher accuracy on the validation set). In the first version, we didn't take into account data augmentation techniques, resulting in overfitting for some of the configurations we have tried. 

In particular, we have tried different values of lr and weight decay and also different schedulers, mantaining the number of epochs equal to 20 in order not to require too much time for executions.  
LR = [0.05, 0.01, 0.005, 0.001]  
WD = [1e-5, 5e-5, 1e-4]  
Schedulers:
- StepLR, gamma = 0.1, step_size = num_epochs//3;
- Cosine annealing, T_max=num_epochs//3, eta_min=1e-4;
- Exponential LR, gamma = 0.9;
- Exponential LR, gamma = 0.95.

In this first setting, the best configuration we obtained was with the StepLR, LR = 0.005 and WD = 5e-5, with a validation accuracy of 41%. The main problem was related to overfitting. Indeed, as it can be observed in the following plot, the training accuracy reached was approximately 0.8, resulting in a huge gap between the two.
![alt text](images_report/image.png)


This is the training loss for all the schedulers with LR = 0.005. LR is probably the most important parameter, that's why we decided to report it:  

![alt text](images_report/image-1.png)

Observing the way the loss was behaving for the StepLR and the cosine annealing schedulers, we also tried to increase the number of epochs before restarting (T_max and step_size), but the result of those experiments are not reported because not meaningful anymore after data augmentation techniques have been introduced. 

We decided to introduce data augmentation techniques to reduce overfitting as reported by the authors of [Reddi et al., Adaptive federated optimization. ICLR 2021], with random crop and random horizontal flips. The configuration that gave us the best result this time is: LR = 0.01, WD = 0.0001 and ExponentialLR (gamma=0.9).  
This is the plot of training accuracy vs validation accuracy, to take into account also overfitting and evaluate if the weight decay should be increased:   

![alt text](images_report/image-2.png)

The problem related to overfitting is clearly solved, but we observe the opposite one: underfitting. In order to make it better, we have to train for longer. This is not a real problem, we did this test with fewer epochs with respect to the ones we were planning to use for the final model used for tests. Also, by plotting losses, we observe that the exponential LR seems to be the best choice because it's the one that decreases more regularly:  
![alt text](images_report/image-3.png)  


So, the previously reported one, will be the final configuration used for tests. For the final training, we decided to set, as a first try, a number of epochs equal to 200 and observe how the model behaves.
