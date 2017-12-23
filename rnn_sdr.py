import copy, numpy as np  #try to adapt so numpy and copy not needed
np.random.seed(0)

class Layer:
    def __init__(self, d_in, d_out):
        self.weights = 2 * np.random.random((d_in, d_out)) - 1  # e.g. 2,3 = [[.0  .0  .0]  [.0  .0  .0]]
        self.updates = np.zeros_like(self.weights)  #weight update values
    
    def f(self, values):
        return np.dot( values, self.weights )    

    def adjust(self, learning_rate):
        self.weights += self.updates * learning_rate

class RNN:
        
    def __init__(self, topology, seq_length):   
        if len(topology) == 3: 
            self.seq_length = seq_length
            [self.d_i, self.d_h, self.d_o] = topology
            self.l0 = Layer(self.d_i, self.d_h)
            self.l1 = Layer(self.d_h, self.d_h)
            self.l2 = Layer(self.d_h, self.d_o)
        else:
            print('''
            the topology provided is in the incorrect format.
            topology must be a list of only 3 integers 
            topology = [input dimension, hidden dimension, output dimension].  
            e.g. RNN([2,4,2])''')

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def d_sigmoid(self, x): 
        return x*(1-x) #convert output of sigmoid function to it derivative
    
    def predict(self, i, remember = False):
        hidden_value = self.sigmoid( self.l0.f(i) + self.l1.f(self.memory[-1]) )
        if remember:    
            self.memory.append(copy.deepcopy(hidden_value))
        return self.sigmoid( self.l2.f(hidden_value) )

    def forward(self, i, o):
        prediction = np.zeros_like(i)
        for n in range(self.seq_length): #going through binary 
            input_bits = np.array([[i[self.seq_length - n - 1]]])
            output_bits = np.array([[o[self.seq_length - n - 1]]]).T
            p = self.predict(input_bits, remember = True)
            e = output_bits - p  #error
            self.deltas.append( e * self.d_sigmoid( p ) )
            self.error += np.abs(e[0])
            prediction[self.seq_length - n - 1] = int(np.round(p))
        return prediction
    
    def backward(self,i):
        hidden_delta_future = np.zeros(self.d_h)
        for n in range(self.seq_length): 
            input_bits = np.array([[i[n]]])
            hidden_value = self.memory[-n-1]
            prev_hidden_value = self.memory[-n-2]
            output_delta = self.deltas[-n-1]

            hidden_delta = (hidden_delta_future.dot( self.l1.weights.T) +  output_delta.dot( self.l2.weights.T )) * self.d_sigmoid( hidden_value)

            self.l2.updates += np.atleast_2d(hidden_value).T.dot(output_delta)
            self.l1.updates += np.atleast_2d(prev_hidden_value).T.dot(hidden_delta)
            self.l0.updates += input_bits.T.dot(hidden_delta)        
            
            hidden_delta_future = hidden_delta
    
    def dump_to_disk(self):
        "store trained weights & memory of network to file"
        pass

    def init_memory(self):
        self.error = 0.0
        self.deltas = []
        self.memory = []
        self.memory.append(np.zeros(self.d_h)) #memory's initial time step is zero since no sequence has came before it
        
        self.l2.updates *= 0
        self.l1.updates *= 0
        self.l0.updates *= 0

    def train(self, training_inputs, training_outputs, errorThresh = 0.01, alpha = 0.1, iterations = 10000, log_rate = 1000):
        for i in range(iterations): #or self.error < errorThresh
            
            self.init_memory()
            p = self.forward(training_inputs[i],training_outputs[i])
            self.backward(training_inputs[i])

            self.l0.adjust(alpha)
            self.l1.adjust(alpha)
            self.l2.adjust(alpha)

            if(i % log_rate == 0):
                print("Input:", str(training_inputs[i]))
                print("Expected:", str(training_outputs[i]))
                print("Predicted:", str(p))
                print("Error:", str(self.error))
                print("------------")
        self.dump_to_disk()


int2binary = {}
binary = np.unpackbits(np.array([range(2**8)],dtype=np.uint8).T,axis=1)
for i in range(2**8):
    int2binary[i] = binary[i] #[{...2:[0 0 0 0 0 0 1 0], ...}]

A, B = [], []
for i in range(100000): #make a quick training set
    a_int = np.random.randint((2**8)/4) # int version
    A.append(int2binary[a_int]) # binary encoding
    b_int = a_int *2
    B.append(int2binary[b_int]) # binary encoding
#-----TEST-------
net = RNN([1,3,1], 8)
net.train(A,B)
