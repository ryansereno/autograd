class Neuron:
  def __init__(self, nin):
    #on creation, initialize neuron with random weights (one for each input)
    #and a random bias
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))

  def __call__(self, x):
    #when the neuron is called, it takes in an input (an array the same size as the neuron input)
    #dot product of weights and input, and tanh activation
    activation = sum((wi*xi for wi, xi in zip(self.w, x)), start=self.b) 
    out = activation.tanh()
    #return a single output scalar
    return out

class Layer:
  def __init__(self, nin, nout):
    #initialize with number of inputs (for ever neuraon) and number of outputs (also the number of neurons)
    self.neurons = [Neuron(nin) for _ in range(nout)]

  def __call__(self,x):
    outs = [n(x) for n in self.neurons]
    return outs

class MLP:
  def __init__(self, nin, nouts):
    #combine nin with nouts
    sz = [nin] + nouts
    #create an array of Layer objects, initialized with the input and output demensions, as defined by sz (the nin and nouts)
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

  def __call__(self, x):
    #pass x through the first layer, then redefined x on output, then pass x into the next layer, recursively
    for layer in self.layers:
      x = layer(x)
    return x
