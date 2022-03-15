import random
import numpy as np
from pathlib import Path
import os

class Network():

    def __init__(self, layersizes):
        """Die Methode "__init__" fungiert ähnlich wie in java als "Konstruktor" der Klasse Network. 
        Hier werden die Attribute wie die Gewichte "weights" und Schwellen "biases" definiert. Dem Konstruktor
        werden die einzelnen größen der Layern als Liste übergeben zum Beispiel (80,89,32,43) Dabei ist die 
        erste Zahl die Anzahl der Input Neuronen die zweiten beiden (bzw. die dazwischen) die Anzahl der 
        Hidden Layer und deren Neuronen, die letzte steht für die Output Neuronen"""
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
        """Die Methode kriegt "training_image" als Input und gibt dann den Wert mit der höchsten Aktivierung zurück,
        also dem für den sich das netzwerk entscheidet"""
        predict = self.calculateOneInput(training_image)
        return np.argmax(predict)

    def calculateOneInput(self, input):
        """Die Methode kriegt "input" als Input und gibt dann die Erbenisse vom letzten Layer zurück, in form eines Arrays"""
        for weight, bias in zip(self.weights, self.biases):
            input = sigmoid(np.dot(weight, input) + bias)
        return input
    
    
    def testdatapackage(self, test_data, test_labels):
        """Die Methode kriegt zwei inputs, zum einen die "test_data", also ein Stapel voller test dateien und einen zweiten,
        genausogroßen Stapel an test_labels, also die zu den testdatendazugehörigen Erbenissen."""
        prediction_list = []
        label_list = []
        #das Netzwerk gibt seine Lösung an, ergebnisse werden gespeichert.
        for input in test_data:
            prediction_list.append(self.predictOneInput(input))
        #höchster Wert der Label, also die korrekten werte werden nochmal zum vergelichen gespeichert.
        for label in test_labels:
            label_list.append(np.argmax(label))
        right_guesses = 0
        for prediction, label in zip(prediction_list, label_list):
            if(prediction == label):
                right_guesses = right_guesses+1
        #rückgabe der von Netzwerk richtigen Ergebnissen.
        return right_guesses

    def saveNetwork(self, network_name):
        """Methode zum sichern eines Netzwerkes. kriegt namen als Input "network_name"."""
        #alle Werte werden an dem namen des Netzwerkes, also im gleichnamigen Ordner gespeichert
        Path('./networks/'+network_name).mkdir(parents=True, exist_ok=True)
        np.save(str('./networks/'+network_name+'/biases.npy'), self.biases)
        np.save(str('./networks/'+network_name+'/weights.npy'), self.weights)
        np.save(str('./networks/'+network_name+'/layersizes.npy'), self.layersize)
        np.save(str('./networks/'+network_name+'/layernumber.npy'), self.layernumber)

    def loadNetwork(self, network_name):
        """Methode zum laden eines Netzwerkes. kriegt namen als Input "network_name"."""
        #lädt alle werte in die Variablen des aktuellen Netzwerkes
        self.biases = np.load(str('./networks/'+network_name+'/biases.npy'), allow_pickle=True)
        self.weights = np.load(str('./networks/'+network_name+'/weights.npy'), allow_pickle=True)
        self.layersize = np.load(str('./networks/'+network_name+'/layersizes.npy'), allow_pickle=True)
        self.layernumber = np.load(str('./networks/'+network_name+'/layernumber.npy'), allow_pickle=True)
    
    
    def stochastic_gradient_descent(self, training_images, training_labels, test_images, test_labels, training_rounds, minibatch_size, learning_rate):
        """Methode zum trainieren des Neuronalen Netzwerkes. Sie bekommt zum einen "training_images" und "training_labels",
        also eimnen stapel gleichgroßer daten zum trainieren des Netzwerkes. Dasselbe gilt für "test_images" und "test_labels".
        Außerdem werden die Runden des Trainings, also die durchläufe angegeben und die learning Rate, die sich Aus der Formel zur
        Gradient descent ergibt."""
        #längen der trainings und testdaten und einen speichern der richtig erratenen Werte
        n = len(training_images)
        n_test = len(test_images)
        roundsstats = []
        #einteilen der trainingsdaten in mini_batches, also kleine Pakete.
        for j in range(training_rounds):
            mini_batches = []
            counter = 0
            for i in range(n//minibatch_size):
                mini_batch = []
                for k in range(minibatch_size):
                    mini_batch.append([(training_images[counter], training_labels[counter])])
                    counter = counter+1
                #mischen der minibatches innerhalb sich selbst. Und hinzufügen zur Liste der mini_batches
                random.shuffle(mini_batch)
                mini_batches.append(mini_batch)
            #für jedes der nur erhaltenen mini_batches die methode update_mini_batch durchführen, zum trainien des Netzwerkes.
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            #ausgeben der Epochendaten, anhand des tests mit den testdaten.
            print("Epoch {} : {} / {}".format(j,self.testdatapackage(test_images, test_labels),n_test))
            roundsstats.append((self.testdatapackage(test_images, test_labels))/n_test)
        return roundsstats

    
    def update_mini_batch(self, mini_batch, learning_rate):
        """Die Methode dient dazu die Backpropagation im Sinne des Gradient Descent durchzuführen. Sie updatet die Gewichte und Schwellen.
        Sie erhälz zum einen die "lerarning_rate", um dann aus den erhaltenen Vektoren durch die Backprop. die Gewichte und Schwellen an zu passen
        und zum anderen das "mini_batch", also ein Paket aus trainings_images und trainingslabels."""
        #Null-Arrays für die angepassten Gewichte und Schwellen in der deselben Größe wie im Originall anlegen.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        #auf jeder Trainingsdata (Trainingsbild und Label) wird zunächst:
        for batch in mini_batch:
            #delta_nable_b und delta_nabla_w erzeugt, sie erhält man durch die Methode "backprop"
            delta_nabla_b, delta_nabla_w = self.backprop(batch[0][0], batch[0][1])
            #delta_nable_b und delta_nabla_w werden jeweils zu nabla_b und nabla_w hinzugefügt
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        #Gewichte und Schwellen werden anhand der Gradient Descent Regel angepasst.
        self.weights = [w-(learning_rate/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, image, right):
        """die methode Backpropagation kriegt sum einen das "image", in form des Vektors übergeben und dann noch 
        die richtige ösung, also das dazugehörige label. ("right")"""
        #Null-Arrays für die angepassten Gewichte und Schwellen in der deselben Größe wie im Originall anlegen.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #speichern der aktivierungen aus dem image und in eine Liste der aktivierungen pro Layer.
        activation = image
        activations = [image]
        #anlegen der z Werte in einer Liste, also die Werte vor der Aktivierungsfunktion
        zs = []
        #für jedes gewicht und jede Schwelle: (Schritt der Backpropagation: 2)
        for b, w in zip(self.biases, self.weights):
            #a.	Normal durch das Netzwerk „füttern“.
            #z-Werte berechnen und zur Liste hinzufügen (nach Layern)
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #b.	Mithilfe von Gleichung 1 im letzten Layer den „Fehler“ einbauen und einen Vektor der neuen Werte ermitteln.
        #Gleichung 1:
        delta = self.cost_derivative(activations[-1], right) * sigmoid_derivation(zs[-1])
        #Gleichung 3:
        nabla_b[-1] = delta
        #Gleichung 4:
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        #c.	Mithilfe von Gleichung 2 den Fehler bis in den ersten Layer bringen
        #loop für jeden Layer
        for l in range(2, self.layernumber):
            #z-Werte vom Layer zwischenspeichern
            z = zs[-l]
            #alle z-Werte durch die Ableitung von der Sigmoidfunktion jagen.
            sp = sigmoid_derivation(z)
            #Gleichung 2:
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            #Gleichung 3:
            nabla_b[-l] = delta
            #Gleichung 4:
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        #zurückgeben der beiden Arrays nabla_b und nabla_w
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """gibt die Ableitung der Kostenfunktion zurück und erhält dafür zum einen "output_activations", also die Aktivierungen des letzten Layers
        und y, also die eigendlichen erwünschten Lösungen."""
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

def doesNetworkExists(network_name):
    """simple Methode zum testen, ob ein Netzwerk existiert anhand des Namens übergeben durch "network_name"."""
    path = './networks/'+network_name
    return os.path.exists(path)
