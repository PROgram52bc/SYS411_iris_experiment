from math import exp
from random import uniform

class Neuron:

    """ Represents a single neuron """
    class Connection:

        """ Represents a connection between two neurons """

        def __init__(self, frm, to, w):
            """ initialize the connection object

            :frm: the Neuron that connects from
            :to: the Neuron that connects to
            :w: the weight between

            """
            if not isinstance(frm, Neuron) or \
                    not isinstance(to, Neuron):
                        raise Exception("invalid frm or to object: must be a Neuron instance")
            self.frm = frm
            self.to = to
            self.w = w
            frm.outputs.append(self)
            to.inputs.append(self)

        def __repr__(self):
            return f"Connection(w={self.w})"

    def __init__(self, activation=0, errdrv=0, normalizer=None):
        self.inputs = []   # a list of Connection objects
        self.outputs = []  # a list of Connection objects
        self.activation = activation
        self.errdrv = errdrv
        self._normalizer = normalizer or (lambda n: 1/(1+exp(-n)))

    def __repr__(self):
        return f"Neuron(activation={self.activation}, errdrv={self.errdrv}, in={len(self.inputs)}, out={len(self.outputs)})"

    def addInput(self, inputNode, weight):
        """ add an input neuron

        :inputNode: the input neuron
        :weight: the weight associated
        :returns: the Connection object

        """
        return self.Connection(inputNode, self, weight)

    def addOutput(self, outputNode, weight):
        """ add an output neuron

        :outputNode: the output neuron
        :weight: the weight associated
        :returns: the Connection object

        """
        return self.Connection(self, outputNode, weight)

    def getWeightedSum(self):
        """ compute the weighted sum from all input nodes
        :returns: the weighted sum

        """
        return sum(map(lambda c: c.frm.activation * c.w, self.inputs))

    def updateActivation(self):
        """ update activation based on the input nodes
        :returns: the new activation value

        """
        self.activation = self._normalizer(self.getWeightedSum())
        return self.activation

    def updateErrDrv(self, targetValue=None):
        """ update the error derivative

        :targetValue: if not None, the error is taken as targetValue-self.activation
                    otherwise, error = weighted sum of all the output nodes' errdrv values
        :returns: the delta calculated by activation * (1-activation) * error

        """
        if targetValue is not None:
            error = targetValue - self.activation
        else:
            error = sum(map(lambda c: c.to.errdrv * c.w, self.outputs))
        a = self.activation

        self.errdrv = a * (1-a) * error
        return self.errdrv


class NeuralNetwork:

    """A NeuralNetwork class that contains layers of neurons and has training capability"""

    def __init__(self, dimension=[1,1], bias=True, rand=True, rate=1):
        """

        :dimension: an array specifying the number of neurons in each layer
        :bias: add a bias neuron for all layers except the input layer, bias is always the first input node
        :rand: set random weights between 0 and 1, otherwise set all weights to 1
        :rate: the learning rate

        """
        if len(dimension) < 2:
            raise Exception("At least two layers required in a neural network")
        self.layers = []
        firstLayerNumNeurons = dimension[0]
        firstLayer = [Neuron() for _ in range(firstLayerNumNeurons)]
        self.layers.append(firstLayer)

        for layerNum, numNeurons in enumerate(dimension[1:], 1):
            layer = []
            # for each neuron in the layer
            for _ in range(numNeurons):
                neuron = Neuron()
                # add bias
                neuron.addInput(Neuron(1), uniform(0,1) if rand else 1)
                # connect to the previous layer
                for inputNeuron in self.layers[layerNum-1]:
                    neuron.addInput(inputNeuron, uniform(0,1) if rand else 1)
                layer.append(neuron)
            self.layers.append(layer)

        self.rate = rate

    @property
    def numInput(self):
        return len(self.layers[0])

    @property
    def numOutput(self):
        return len(self.layers[-1])

    def validateDataIO(self, inputData):
        if len(inputData) != self.numInput + self.numOutput:
            raise ValueError(f"data for training has an invalid length {len(inputData)}. Expected {self.numInput + self.numOutput}")

    def validateDataI(self, inputData):
        if len(inputData) != self.numInput:
            raise ValueError(f"data for prediction has an invalid length {len(inputData)}. Expected {self.numInput}")

    def validateDataO(self, inputData):
        if len(inputData) != self.numOutput:
            raise ValueError(f"data for back propagation has an invalid length {len(inputData)}. Expected {self.numOutput}")

    def getDataI(self, inputData):
        return inputData[:self.numInput]

    def getDataO(self, inputData):
        return inputData[self.numInput:]

    def __repr__(self):
        s = "NeuralNetwork(\n"
        for i, layer in enumerate(self.layers):
            s += f"\tLayer {i}:\n"
            for neuron in layer:
                s += f"\t\t[activation: {neuron.activation:.4f}, errdrv: {neuron.errdrv:.4f}, i: {len(neuron.inputs)}, o: {len(neuron.outputs)}\n"
                s += "\t\t\ti: " + ",".join(map(lambda i: "{:.4f}".format(i.w), neuron.inputs)) + "\n"
                s += "\t\t\to: " + ",".join(map(lambda i: "{:.4f}".format(i.w), neuron.outputs)) + "\n"
                s += "\t\t]\n"
        s += ")"
        return s

    def setLayerActivations(self, layerNum, activations):
        """ set the activation of the specified layer, mainly used to set the first layer

        :layerNum: the specific layer to be set
        :activations: a list of numbers, the rest are ignored if length is less than the layer's number of neurons,
                    use None to indicate no change
        :returns: None

        """
        for neuron, activation in zip(self.layers[layerNum], activations):
            if activation is not None:
                neuron.activation = activation

    def setInputValues(self, activations):
        """ a wrapper around setLayerActivations

        :activations: a list of numbers, the rest are ignored if length is less than the layer's number of neurons,
        :returns: None

        """
        self.setLayerActivations(0, activations)

    def setWeights(self, layerNum, nodeNum, weights, setOutput=False):
        """ set the weights of the specified node

        :layerNum: the specific layer to be set
        :nodeNum: the specific neuron in the layer
        :weights: a list of numbers, the rest are ignored if length is less than the node's number of connections
                    use None to indicate no change
        :setOutput: set the input weights by default, if True, set output instead
        :returns: None

        """
        node = self.layers[layerNum][nodeNum]
        connections = node.outputs if setOutput else node.inputs
        for connection, weight in zip(connections, weights):
            if weight is not None:
                connection.w = weight

    def forwardPropagate(self, inputValues):
        """ perform forward propagation based on the input values

        :inputValues: the list of input values to be used
        :returns: None

        """
        self.validateDataI(inputValues)
        self.setInputValues(inputValues)
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.updateActivation()

    def backPropagate(self, targetValues):
        """ perform backward propagation based on the target values

        :targetValues: the list of target values to be used
        :returns: None

        """
        self.validateDataO(targetValues)
        for targetValue, neuron in zip(targetValues, self.layers[-1]):
            neuron.updateErrDrv(targetValue)

        # update the errdrv of all layers using the old weight
        for layer in reversed(self.layers[:-1]):
            for neuron in layer:
                neuron.updateErrDrv()

        # update the weights in all layers
        for layer in reversed(self.layers):
            for neuron in layer:
                for connection in neuron.inputs:
                    connection.w += self.rate * connection.frm.activation * neuron.errdrv

    def train(self, data):
        """ train the network

        :data: values of input and target output
        :returns: None

        """
        self.forwardPropagate(self.getDataI(data))
        self.backPropagate(self.getDataO(data))

    def predict(self, inputData):
        """ use the network to make prediction

        :inputData: TODO
        :returns: TODO

        """
        self.forwardPropagate(inputData)
        return [ n.activation for n in self.layers[-1] ]

    def computeError(self, inputData):
        """TODO: Docstring for computeError.

        :inputData: the data containing input and output
        :returns: the error as a floating point number

        """
        predicted = self.predict(self.getDataI(inputData))
        target = self.getDataO(inputData)
        return sum([ (a-b)**2 for a, b in zip(predicted, target) ])


def main():
    #################################################
    #  Training a network to predict the XOR logic  #
    #################################################

    # create a new neural network
    network = NeuralNetwork([2,2,2], rate=1)

    # set the weights

    # Original weights
    network.setWeights(1, 0, [.1, .2, .4])
    network.setWeights(1, 1, [-.1, -.3, .3])
    network.setWeights(2, 0, [-.2, .3, .5])
    network.setWeights(2, 1, [.3, -.2, -.4])

    # # Weights that perform well
    # network.setWeights(1, 0, [0.9845,0.2826,0.3580])
    # network.setWeights(1, 1, [0.6534,0.6078,0.5237])
    # network.setWeights(2, 0, [0.0505,0.4760,0.5363])
    # network.setWeights(2, 1, [0.8470,0.5834,0.2911])

    print("Network initial state:")
    print(network)

    # propagate
    dataSets = [
            [1,1,0,1],
            [1,0,1,0],
            [0,1,1,0],
            [0,0,0,1]
            ]

    # # debug test
    # network.forwardPropagate([1,1])
    # network.backPropagate([0,1])
    # print(network)

    for epoch in range(500000):
        sumError = 0
        for data in dataSets:
            network.train(data)
            sumError += network.computeError(data)
        if epoch % 10000 == 0:
            print(f"epoch: {epoch}, sumError: {sumError}")
    for data in dataSets:
        inputData = network.getDataI(data)
        result = network.predict(inputData)
        print("input:", inputData, "result:", result)

if __name__ == "__main__":
    main()
