import pprint
import csv
import itertools
import random
from .fragmentation import stratify, fragment

class Adapter:
    """ An abstract class that adapts input data to a neural network compatible input data """
    @staticmethod
    def adapt(rawData):
        """ takes in raw data and transform to the desired format

        :rawData: custom raw data
        :returns: specified data format suitable for neural network training

        """
        raise NotImplementedError("Must implement the adapt method in subclass")

class IrisDataRotationAdapter(Adapter):
    @staticmethod
    def transformRow(row):
        newRow = row[1:-1] # remove the first and last item
        newRow = [ float(item) for item in newRow ] # cast into float
        m = {
            "Iris-setosa"     : [ 1, 0, 0 ],
            "Iris-versicolor" : [ 0, 1, 0 ],
            "Iris-virginica"  : [ 0, 0, 1 ]
        }
        newRow += m[row[-1]] # unpack the last item as integers
        return newRow

    @staticmethod
    def adapt(rawData):
        """ takes in the raw Iris CSV data, transforms into 5 rotations
        additionally, the following tasks are performed:
        - casting input strings into floating point numbers
        - converting the output string values into three unpacked integer values, where
            Iris-setosa         =>   1 0 0
            Iris-versicolor     =>   0 1 0
            Iris-virginica      =>   0 0 1
        - the identification number in the first column is removed

        For example, a row like the following in the input data
        ['1', '5.1', '3.5', '1.4', '0.2', 'Iris-setosa']
        will be transformed into:
        [ 5.1, 3.5, 1.4, 0.2, 1, 0, 0 ]

        :rawData: the rows in Iris data
        :returns: a 60-20-20 split with 5 rotations that each data is equally represented in training, cv, and test.
                the data is returned as an iterable of dictionaries, each dictionary contains 'training', 'cv', and 'test' keys.
        """
        data = rawData
        data = stratify(data, lambda c:c[5]).values() # stratify according to the category name (e.g. Iris-setosa) at index 5
        data = [ fragment(category, 5) for category in data ] # randomly fragment each category into 5 sections
        data = [ list(itertools.chain(*sections)) for sections in zip(*data) ] # join each section across the categories

        # 5 rotations
        for _ in range(5):
            rotation = {
                "training": list(itertools.chain(*data[:3])),
                "cv": data[3],
                "test": data[4]
            }

            # shuffle
            for i in rotation:
                rotation[i] = [ IrisDataRotationAdapter.transformRow(row) for row in rotation[i] ] # transform the rows
                random.shuffle(rotation[i])

            yield rotation

            # rotate
            data.insert(0, data.pop())

# def main():
#     pp = pprint.PrettyPrinter(indent=2)
#     with open("../data/Iris.csv") as f:
#         reader = csv.reader(f)
#         next(reader) # skip first line
#         data = list(reader)
#     data = IrisDataRotationAdapter.adapt(data)
#     for i, rotation in enumerate(data):
#         print("Rotation: {}".format(i))
#         pp.pprint(rotation)

# if __name__ == "__main__":
#     main()
