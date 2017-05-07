import math

class Vector:
    def __init__(self, it):
        self.v = [x for x in it]

    def __len__(self):
        return len(self.v)

    def __getitem__(self, key):
        return self.v[key]

    def __setitem__(self, key, item):
        self.v[key] = item

    def __iter__(self):
        return iter(self.v)

    def __mul__(self, other):
        if isinstance(other, Vector):
            # let's dot it
            if len(self) != len(other):
                raise ArithmeticError(
                    "Cannot dot vector%d with vector%d"%(len(self), len(other)))
            # dot dot
            t = 0
            for i in range(len(self)):
                t += self[i] * other[i]
            return t
        if type(other) in (int, float):
            # let's scalar multiply it
            t = []
            for i in self:
                t.append(i * other)
            return t
        raise TypeError("Cannot multiply Vector with %s"%type(other))

    def __repr__(self):
        return "(" + ",".join([str(n) for n in  self.v]) + ")"

def g(x):
    return 1 / (1 + math.exp(-x))

def neuron(activations, weights):
    return g(activations * weights)


# size is [upper layer, lower layer] not counting bias
# Calculates the result between two layers.
# weights is a list of Vectors. Each vector is a row.
def doLayer(x, weights, bias=False):
    if bias:
        x.v = [1] + x.v # creates a new list
    out = [None for thing in range(len(weights))]
    for i in range(len(weights)):
        out[i] = g(x * weights[i])
    return Vector(out)

# len of allweights, sizes, biases must be the same.
def doNeuralNetwork(x, allweights, biases):
    intermediate = None
    for i in range(len(allweights)):
        if i == 0:
            intermediate = doLayer(x, allweights[i], biases[i])
        else:
            intermediate = doLayer(intermediate, allweights[i], biases[i])
    return intermediate

def classifyDigit(x, allweights):
    result = doNeuralNetwork(x, allweights, [True, True])
    #print(result)
    max_i = 0 # the index of the max
    max_n = result[0] # the value of the max
    for i in range(1, 10):
        if result[i] > max_n:
            max_i = i
            max_n = result[i]
    return max_i + 1

def vectorsFromCSV(filename):
    out = []
    with open(filename) as f:
        raw = f.read()[:-1].split("\n")
    for line in raw:
        numbers = [float(n) for n in line.split(",")]
        out.append(Vector(numbers))
    return out

def intsFromFile(filename):
    out = []
    with open(filename) as f:
        raw = f.read()[:-1].split("\n")
    for line in raw:
        out.append(int(line))
    return out

def validate(inputs, labels, hidden, output, noisy=False):
    errors = 0
    for i in range(len(inputs)):
        d = classifyDigit(inputs[i], [hidden, output])
        if d != labels[i]:
            errors += 1
        if noisy and i % 500 == 0:
            print(i, errors / (i+1))
    return errors / len(inputs)

def validateInRange(inputs, labels, hidden, output, r, noisy=False):
    errors = 0
    inputs = inputs[r[0]:r[1]]
    labels = labels[r[0]:r[1]]
    for i in range(len(inputs)):
        d = classifyDigit(inputs[i], [hidden, output])
        if d != labels[i]:
            errors += 1
            print(d, labels[i])
        if noisy and i % 500 == 0 and i != 0:
            print(i, errors / (i+1))
    return errors / len(inputs)


DO_NN = True
if __name__=="__main__":
    data   = vectorsFromCSV("ps5_data.csv")
    labels = intsFromFile("ps5_data-labels.csv")
    hidden = vectorsFromCSV("ps5_theta1.csv")
    output = vectorsFromCSV("ps5_theta2.csv")

    # Network is 400 -> 25 -> 10
    if DO_NN:
        errorrate = validate(data, labels, hidden, output, noisy=True)
        #errorrate = validateInRange(data, labels, hidden, output, [500, 1000], noisy=True)
        print("Error rate: %.1f" % (errorrate * 100))