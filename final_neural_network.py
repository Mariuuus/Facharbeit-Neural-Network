from itertools import count
import random
import numpy as np
from pathlib import Path

class Network():
    """Die Methode "__init__" fungiert ähnlich wie in java als "Konstruktor" der Klasse Network. 
    Hier werden die Attribute wie die Gewichte "weights" und Schwellen "biases" definiert. Dem Konstruktor
    werden die einzelnen größen der Layern als Liste übergeben zum Beispiel (80,89,32,43) Dabei ist die 
    erste Zahl die Anzahl der Input Neuronen die zweiten beiden (bzw. die dazwischen) die Anzahl der 
    Hidden Layer und deren Neuronen, die letzte steht für die Output Neuronen"""

    def __init__(self, layersizes):
        #durch self. fungieren die biases und weights als attribute der Klasse
        self.layersize = layersizes
        self.layernumber = len(layersizes)
        #Biases "Schwellen". existieren für alle Layer außer den Input also für jedes "nicht" input Neuron
        self.biases = np.array([np.random.randn(y, 1) for y in layersizes[1:]], dtype=object)
        #Weights "Gewichte". Also Verbindungen der einzelnen Layer/Neuronen miteinander. es gibt soviele, wie es Verbindungen zwischen den Neuronen gibt
        #Kann man sich vorstellen als Array mit Listen für jeden Layer, in den sich eine Liste für die Verbindungen jedes Neurones befindet
        #Visualisierung:
        """ 
                                ----LAYERSIZE (2,4,3)----

        Liste der Gewichte der Input Neuronen:
                       Neuron 1:    Neuron 2:
        Verbindung 1:[ 1.11021248  0.03956981]
        Verbindung 2:[ 0.27766621 -0.94367979]
        Verbindung 3:[-0.74698688 -0.741232  ]
        Verbindung 4:[ 0.45043321  0.70044881]

        Liste der Gewichte des zweiten Layern der Neuronen:
                       Neuron 1:    Neuron 2:   Neuron 3:   Neuron 4:
        Verbindung 1:[-0.51515155  0.96745075  0.16363184  0.81712242]
        Verbindung 2:[ 0.58409127  0.03106383 -1.03246325  0.42714855]
        Verbindung 3:[ 0.03613319  0.42597822  2.37575331 -1.34904226]
        """
        self.weights = np.array([np.random.randn(y, x)for x, y in zip(layersizes[:-1], layersizes[1:])], dtype=object)

    def predictOneInput(self, training_image):
        predict = self.calculateOneNeuron(training_image)
        return np.argmax(predict)

    def calculateOneNeuron(self, input):
        for weight, bias in zip(self.weights, self.biases):
            input = sigmoid(np.dot(weight, input) + bias)
        return input

    def testdatapackage(self, test_data, test_labels):
        prediction_list = []
        label_list = []
        for input in test_data:
            prediction_list.append(self.predictOneInput(input))
        for label in test_labels:
            label_list.append(np.argmax(label))
        right_guesses = 0
        for prediction, label in zip(prediction_list, label_list):
            if(prediction == label):
                right_guesses = right_guesses+1
        return right_guesses
    
    def saveNetwork(self, network_name):
        Path('./networks/'+network_name).mkdir(parents=True, exist_ok=True)
        np.save(str('./networks/'+network_name+'/biases.npy'), self.biases)
        np.save(str('./networks/'+network_name+'/weights.npy'), self.weights)
        np.save(str('./networks/'+network_name+'/layersizes.npy'), self.layersize)
        np.save(str('./networks/'+network_name+'/layernumber.npy'), self.layernumber)

    def loadNetwork(self, network_name):
        self.biases = np.load(str('./networks/'+network_name+'/biases.npy'), allow_pickle=True)
        self.weights = np.load(str('./networks/'+network_name+'/weights.npy'), allow_pickle=True)
        self.layersize = np.load(str('./networks/'+network_name+'/layersizes.npy'), allow_pickle=True)
        self.layernumber = np.load(str('./networks/'+network_name+'/layernumber.npy'), allow_pickle=True)
    
    def stochastic_gradient_descent(self, training_images, training_labels, test_images, test_labels, training_rounds, minibatch_size, learning_rate):

        n = len(training_images)
        n_test = len(test_images)

        for j in range(training_rounds):
            mini_batches = []
            counter = 0
            for i in range(n//minibatch_size):
                mini_batch = []
                for k in range(minibatch_size):
                    mini_batch.append([(training_images[counter], training_labels[counter])])
                    counter = counter+1
                random.shuffle(mini_batch)
                mini_batches.append(mini_batch)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            print("Epoch {} : {} / {}".format(j,self.testdatapackage(test_images, test_labels),n_test))

    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for batch in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(batch[0][0], batch[0][1])
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(learning_rate/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, image, right):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = image
        activations = [image] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], right) * \
            sigmoid_derivation(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.layernumber):
            z = zs[-l]
            sp = sigmoid_derivation(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

def sigmoid(x):
    """
    Methode "sigmoid" fungiert als die Sigmoid Funktion
    Parameter: x als Eingabe des x Wertes für die Funktion 
    gibt den Funktionswert zurück
    """
    return 1.0/(1.0+np.exp(-x))

def sigmoid_derivation(x):
    """
    Methode "sigmoid_derivation" fungiert als die Ableitung der Sigmoid Funktion
    Parameter: x als Eingabe des x Wertes für die Funktion 
    gibt den Funktionswert zurück
    """
    return sigmoid(x)*(1-sigmoid(x))