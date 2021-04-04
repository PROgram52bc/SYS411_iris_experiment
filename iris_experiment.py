import pprint
import csv
import itertools
from statistics import mean
from utils import NeuralNetwork, IrisDataRotationAdapter

DEBUG_LEVEL = 5


def D(level, msg):
    if DEBUG_LEVEL >= level:
        print("[{}] {}".format(level, msg))

def main():
    pp = pprint.PrettyPrinter(indent=2)
    with open("data/Iris.csv") as f:
        reader = csv.reader(f)
        next(reader) # skip first line
        data = list(reader)
    data = IrisDataRotationAdapter.adapt(data)
    data = list(data)
    D(1, "Successfully adapted data")

    # for dataset in five dataset rotations:
    lRates = [0.02, 0.2, 1]
    nNodes = range(2,8)
    nIteration = 3
    maxEpoch = 10

    # record the errors
    errors = {}
    configs = list(itertools.product(lRates, nNodes))
    for rotation, dataset in enumerate(data):
        D(2, f"Rotation {rotation}")
        for config, (lRate, nNode) in enumerate(configs):
            D(3, "config: {}, lRate: {}, nNode: {}".format(config, lRate, nNode))
            errors.setdefault(config, [])

            for iteration in range(nIteration):
                D(4, f"iteration {iteration}")

                nn = NeuralNetwork([4, nNode, 3], rate=lRate)
                # Training
                for epoch in range(maxEpoch):
                    nn.batchTrain(dataset["training"])
                    if epoch % (maxEpoch//10) == 0:
                        error = nn.batchComputeError(dataset["cv"])
                        D(5, f"Epoch: {epoch}, error: {error}")

                # Record error
                error = nn.batchComputeError(dataset["cv"])
                D(4, f"cv error: {error}")
                errors[config].append(error)
    D(1, "Raw errors:")
    pp.pprint(errors)
    for config in errors:
        errors[config] = mean(errors[config])
    D(1, "Average Error:")
    pp.pprint(errors)

    minError = min(errors.items(), key=lambda item:item[1])
    optConfig = minError[0]
    D(1, f"Optimum config: {optConfig} with error: {minError[1]}")

    D(1, f"Retrain network with optimum config: {configs[optConfig]}")
    optErrors = []
    for rotation, dataset in enumerate(data):
        D(2, f"Rotation {rotation}")
        lRate, nNode = configs[optConfig]
        for iteration in range(nIteration):
            D(3, f"iteration {iteration}")
            nn = NeuralNetwork([4, nNode, 3], rate=lRate)
            # Training
            for epoch in range(maxEpoch):
                nn.batchTrain(dataset["training"])
                if epoch % (maxEpoch//10) == 0:
                    error = nn.batchComputeError(dataset["test"])
                    D(4, f"Epoch: {epoch}, error: {error}")

            # Record error
            error = nn.batchComputeError(dataset["test"])
            D(3, f"test error: {error}")
            optErrors.append(error)
    D(1, f"Final error: {mean(optErrors)}")

if __name__ == "__main__":
    main()
