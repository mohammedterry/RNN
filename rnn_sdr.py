import copy, numpy as np  #try to adapt so numpy and copy not needed
np.random.seed(0)

class Layer:
    memory = [] #this is only used if this is the hidden layer
    deltas = []
    
    def __init__(self, d_in, d_out):
        self.weights = 2 * np.random.random((d_in, d_out)) - 1  # e.g. 2,3 = [[.0  .0  .0]  [.0  .0  .0]]
        self.updates = np.zeros_like(self.weights)  #weight update values
        self.memory.append(np.zeros(d_in)) #memory's initial time step is zero since no sequence has came before it
    
    def f(self, values):
        return np.dot( values, self.weights )    

    def adjust(self, learning_rate):
        self.weights += self.updates * learning_rate

class RNN:
    error = 0
    def __init__(self, topology):   
        if len(topology) == 3: 
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
        hidden_value = self.sigmoid( self.l0.f(i) + self.l1.f(self.l1.memory[-1]) )
        if remember:    
            self.l1.memory.append(copy.deepcopy(hidden_value))
        return self.sigmoid( self.l2.f(hidden_value) )

    def forward(self, i, o, bit_length):
        for n in range(bit_length,0,-1): #going through binary 
            input_bits = np.array([[i[n-1]]])
            output_bits = np.array([[o[n-1]]]).T
            p = self.predict(input_bits, remember = True)
            e = output_bits - p  #error
            self.l2.deltas.append( e * self.d_sigmoid( p ) )
            self.error += np.abs(e[0])
        return p
    

    def backward(self,i,bit_length):
        hidden_delta_future = np.zeros(self.d_h)
        for n in range(bit_length): 
            input_bits = np.array([[i[n]]])
            hidden_value = self.l1.memory[-n-1]
            prev_hidden_value = self.l1.memory[-n-2]
            output_delta = self.l2.deltas[-n-1]

            hidden_delta = (hidden_delta_future.dot( self.l1.weights.T) +  output_delta.dot( self.l2.weights.T )) * self.d_sigmoid( hidden_value)

            self.l2.updates += np.atleast_2d(hidden_value.T.dot(output_delta))
            self.l1.updates += np.atleast_2d(prev_hidden_value).T.dot(hidden_delta)
            self.l0.updates += input_bits.T.dot(hidden_delta)        
            
            hidden_delta_future = hidden_delta
    
    def dump_to_disk(self):
        "store trained weights & memory of network to file"
        pass

    def train(self, training_inputs, training_outputs, alpha = 0.1, iterations = 10000, log_rate = 1000):
        for j in range(iterations):
            p = self.forward(training_inputs[j],training_outputs[j], 8)
            self.backward(training_inputs[j], 8)
            self.l0.adjust(alpha)
            self.l1.adjust(alpha)
            self.l2.adjust(alpha)
            if(j % log_rate == 0):
                prediction = ''
                for bit in range(8): # [[0.3312]]
                    prediction += str(int(np.round(p[0][0])))
                print("Input:", str(training_inputs[j]))
                print("Expected:", str(training_outputs[j]))
                print("Predicted:", prediction)
                print("Error:", str(self.error))
                print("------------")
        self.dump_to_disk()


int2binary = {}
binary = np.unpackbits(np.array([range(2**8)],dtype=np.uint8).T,axis=1)
for i in range(2**8):
    int2binary[i] = binary[i] #[{...2:[0 0 0 0 0 0 1 0], ...}]

A = []
a_int = 0 
for i in range(10): #make a quick training set
    a_int = np.random.randint((2**8)/2) # int version
    A.append(int2binary[a_int]) # binary encoding
A *= 50
#-----TEST-------
net = RNN([1,10,1])
net.train(A,A, iterations = 500, log_rate = 100)




