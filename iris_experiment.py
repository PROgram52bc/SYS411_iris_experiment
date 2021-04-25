import pprint
import csv
import itertools
from statistics import mean
from utils import NeuralNetwork, IrisDataRotationAdapter

DEBUG_LEVEL = 5
FLUSH_LEVEL = 3

def D(level, msg):
    if DEBUG_LEVEL >= level:
        print("[{}] {}".format(level, msg), flush=FLUSH_LEVEL>=level)


def runConfig(data, trainingKey="training", testKey="cv", lRate=0.2, nNode=2, nIteration=1, maxEpoch=100):
    """ run iris experiment with the specified configuration

    :data: in the form [ {"cv": [...], "test": [...], "training": [...]}, {...}, ... ]
    :trainingKey: the key to obtain the training data in each dataset
    :testKey: the key to obtain the test data in each dataset
    :lRate: learning rate
    :nNode: number of nodes in the hidden layer
    :nIteration: number of iterations over which to average the errors and accuracies
    :maxEpoch: the maximum number of epochs
    :returns: the error and accuracy as a tuple

    """
    # record the errors
    errors = []
    accuracies = []
    for rotation, dataset in enumerate(data):
        D(2, f"Rotation {rotation}")
        for iteration in range(nIteration):
            D(3, f"iteration {iteration}")
            nn = NeuralNetwork([4, nNode, 3], rate=lRate)
            # Training
            for epoch in range(maxEpoch):
                nn.batchTrain(dataset[trainingKey])
                if epoch % (maxEpoch//10) == 0:
                    error = nn.batchComputeError(dataset[testKey])
                    D(4, f"Epoch: {epoch}, error: {error}")

            # Record error
            error = nn.batchComputeError(dataset[testKey])
            accuracy = nn.batchComputeAccuracy(dataset[testKey])
            D(3, f"error: {error}, accuracy: {accuracy}")
            errors.append(error)
            accuracies.append(accuracy)
    error = mean(errors)
    accuracy = mean(accuracies)
    return error, accuracy


def main():
    pp = pprint.PrettyPrinter(indent=2)
    with open("data/Iris.csv") as f:
        reader = csv.reader(f)
        next(reader) # skip first line
        data = list(reader)
    data = IrisDataRotationAdapter.adapt(data)
    data = list(data)
    D(1, "Successfully adapted data")

    # constants
    nIteration = 3
    maxEpoch = 10000

    # generate config parameters
    # configs = [ {"lRate": 0.02, "nNode": 2}, {"lRate": 0.02, "nNode": 3}, ... ]
    lRates = [0.02, 0.2, 1]
    nNodes = range(2,8)
    tConfigKey = ("lRate", "nNode")
    tConfigVals = itertools.product(lRates, nNodes)
    configs = [ dict(zip(tConfigKey, tConfigVal)) for tConfigVal in tConfigVals ]

    # initiate errors and accuracies for different configurations
    errors = {}
    accuracies = {}
    for i, config in enumerate(configs):
        errors.setdefault(i, [])
        accuracies.setdefault(i, [])
        D(2, "config number: {}\nconfig: {}".format(i, config))
        error, accuracy = runConfig(data, nIteration=nIteration, maxEpoch=maxEpoch, **config)
        errors[i] = error
        accuracies[i] = accuracy

    print("configs:")
    pprint.pprint(configs)
    print("errors:")
    pprint.pprint(errors)
    print("accuracies:")
    pprint.pprint(accuracies)

    optErrorIdx, optError = min(errors.items(), key=lambda item:item[1])
    optAccuracyIdx, optAccuracy = max(accuracies.items(), key=lambda item:item[1])
    D(1, f"Optimum config according to error: {optErrorIdx}, error: {optError}")
    D(1, f"Optimum config according to accuracy: {optAccuracyIdx}, accuracy: {optAccuracy}")

    D(1, f"Retrain network with optimum configs: {optErrorIdx} and {optAccuracyIdx}")

    error, accuracy = runConfig(data, testKey="test", nIteration=nIteration, maxEpoch=maxEpoch, **configs[optErrorIdx])
    D(1, f"config: {optErrorIdx}, error: {error}, accuracy: {accuracy}")
    if optErrorIdx != optAccuracyIdx:
        error, accuracy = runConfig(data, testKey="test", nIteration=nIteration, maxEpoch=maxEpoch, **configs[optAccuracyIdx])
        D(1, f"config: {optAccuracyIdx}, error: {error}, accuracy: {accuracy}")

if __name__ == "__main__":
    main()
