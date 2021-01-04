#importing necessary libraries
import matplotlib.pyplot as plt # for plotting
from dataloader import * # to load the data

# load the data using the dataloader written by Jian Kang
from dataloader import *
dataloader = DataLoader(Xtrainpath='data/train-images-idx3-ubyte.gz',
                        Ytrainpath='data/train-labels-idx1-ubyte.gz',
                        Xtestpath='data/t10k-images-idx3-ubyte.gz',
                        Ytestpath='data/t10k-labels-idx1-ubyte.gz')
Xtrain, Ytrain, Xtest, Ytest = dataloader.load_data()

# flatten each image of size 28x28 pixels into a feature vector
# each value in the feature vectors is divided by 255 to make it in the range [0,1]
Xtrain_feature = np.reshape(Xtrain, (60000, 784)) / 255
Xtest_feature = np.reshape(Xtest, (10000, 784)) / 255
# image is positive (Y = 1) if the digit is 0, else its negative (Y = 0)
Ytrain = Ytrain == 0
Ytrain = Ytrain.astype(int)
Ytest = Ytest == 0
Ytest = Ytest.astype(int)
# hyperparameter settings
num_iterations = 100
learning_rate = 0.1/60000

# plot testing accuracy and log likelihoods vs number of iterations
def plot(y_values, y_value_name):
    x = range(1,num_iterations+1,1)
    plt.plot(x, y_values)
    plt.title(y_value_name + " vs. number of iterations")
    plt.xlabel("number of iterations")
    plt.ylabel(y_value_name)
    plt.show()

# logistic regression using gradient ascent algorithm
def lr_using_ga():
    # initialize w as w = 0 (vector filled with 0)
    w = np.zeros(784)
    # need to obtain testing accuracy for each iteration
    testing_accuracies = []
    # need to obtain log likelihood for each iteration
    log_likelihoods = []
    # gradient ascent algorithm
    for i in range(num_iterations):
        # updating weights
        Ytrain_predictions = 1 / (1 + np.exp(-np.dot(w, Xtrain_feature.T)))
        gradient = np.dot(Xtrain_feature.T, (Ytrain - Ytrain_predictions))
        w += (learning_rate * gradient)
        # getting testing accuracy for current iteration
        predictions = np.round(1 / (1 + np.exp(-np.dot(w, Xtest_feature.T))))
        accuracy = (predictions == Ytest).sum().astype(float) / len(Ytest)
        testing_accuracies.append(accuracy)
        # getting log likelihood for current iteration
        log_likelihood = np.sum(Ytest * np.dot(w, Xtest_feature.T) - np.log(1 + np.exp(np.dot(w, Xtest_feature.T))))
        log_likelihoods.append(log_likelihood)
    #plot testing accuracy vs. number of iterations
    plot(testing_accuracies, "testing accuracy")
    #plot log likelihood vs. number of iterations
    plot(log_likelihoods, "log likelihood")

# run the program
lr_using_ga()
