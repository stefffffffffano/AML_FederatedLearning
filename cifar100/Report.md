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

For the federated setting, the tests for hyperparamter tuning brought us to this configuration: lr=0.1 and wd = 0.001, reaching a final accuracy of 47.81%. Here we report training and validation accuracies over 2000 rounds:  

![alt text](images_report/image-2.png)  

While here are the losses:    


![alt text](images_report/image-3.png)


