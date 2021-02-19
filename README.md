# MNIST-Fashion-Image-Classifier
Train/test data from https://www.kaggle.com/zalando-research/fashionmnist

This is an image classifier that is built in Python using the sci-kit learn package, with numpy for matrix manipulations and pandas for CSV parsing.
The classification goal is to identify a grayscale image of a fashion item into one of ten specific categories:

0: T-shirt
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot

Note that the classification labels for each training example were originally labeled with integeres valued 0-9, and one-hot encoding was used to transform
these labels to a vector of size 10. Additionally, pixel features were scaled via mean normalization using the StandardScaler module in sklearn in an attempt
to speed up convergence.

The train dataset consisted of 40,000 examples while the hold-out dataset consisted of 20,000 examples.

The model used to accomplish this task is a multi-layer perceptron, the MLPClassifier module in sci-kit learn, sporting an input layer of 784 features (pixels),
a single hidden layer of 350 nodes, and an output layer of ten nodes, representing the classification label. The activation function used was the default 'relu'
activation offered by sklearn, and the 'adam' solver was used to perform the gradient descent and learning of weights. In order to tune hyperparameters like 
batch-size, initial learning rate, and the regularization parameter, the GridSearchCV module of sklearn was used, eventually settling on parameters of:

alpha (regularization) of 0.001
initial learning rate of 0.001
mini-batch size of 128

The full details of performance can be found below. Not shown are the accuracy and AUC score, which were 0.86 and 0.936 respectively (performance on hold-out dataset).


![Alt text](https://github.com/rshbabdulvahid/MNIST-Fashion-Image-Classifier/blob/master/fashion_classification.PNG) 


