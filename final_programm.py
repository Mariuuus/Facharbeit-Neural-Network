from random import randint
import numpy as np
from final_neural_network import Network
import final_neural_network as network
import matplotlib.pyplot as plot
import tkinter as tk
from matplotlib import pyplot as plt

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

nn = Network([784, 30, 10, 10])

def trainNetwork():
    print("Was möchtest du machen?")
    print(' - erstelle ein neues Netwzwerk - tippe "create"')
    print(' - traniere ein Netwerk - tippe "train"')
    action = input('"train" or "create":')
    if action == 'train':
        networkname = input('Name des Netzwerkes, dass du trainieren möchtest:')
        if network.doesNetworkExists(networkname):
            nn.loadNetwork(networkname)
            root = tk.Tk()
            tk.Label(root, text=('Netzwerk:',networkname), font=("Helvetica", 20), justify='center', pady=10).grid(row=0, column=0)
            tk.Label(root, text="Runden/Epochen").grid(row=1)
            tk.Label(root, text="minibatch-Größe").grid(row=2)
            tk.Label(root, text="lern-Rate").grid(row=3)
            e1 = tk.Entry(root)
            e2 = tk.Entry(root)
            e3 = tk.Entry(root)
            e1.grid(row=1, column=1)
            e2.grid(row=2, column=1)
            e3.grid(row=3, column=1)

            tk.Button(root, text="trainieren", font=("Helvetica", 16)).grid(row=4, column=0, padx=10)
            tk.Button(root, text="schließen", font=("Helvetica", 16), command=root.destroy).grid(row=4, column=1, padx=10)

            root.mainloop()
        else:
            print('Dieses Netwerk existiert nicht!')
    elif action == 'create':
        networkname = input('Name des Netzwerkes, dass du erstellen möchtest:')
        if not network.doesNetworkExists(networkname):
            nn.saveNetwork(networkname)
            root = tk.Tk()
        else:
            print('Name bereits vergeben!')
    else:
        print('Keine Richtige Eingabe!')

trainNetwork()

def useNetwork():
    networkname = input('Name des Netzwerkes, dass du visuell testen möchtest:')

    #nn.saveNetwork("testnet")
    nn.loadNetwork(networkname)
    nn.stochastic_gradient_descent(training_images, training_labels, test_images, test_labels, 5, 10, 3.0)
    #nn.saveNetwork(networkname)

    #print(nn.testdatapackage(test_images, test_labels))
    #nn.saveNetwork(networkname)

    root = tk.Tk()

    def predict_new_picture():
        #net.SGD(training_data, 1, 10, 3.0, test_data=test_data)
        randompic = randint(0, 10000)
        plt.imshow(training_images[randompic].reshape(28,28), cmap='gray')
        path = './img/'+"current"+'.png'
        plt.savefig(path, dpi=100)
        text1 = "Prediction: "+ str(nn.predictOneInput(training_images[randompic]))
        text2 = "Right Answer: "+ str(np.argmax(training_labels[randompic]))
        label1.config(text=text1)
        label2.config(text=text2)
        bild1.config(file=path)
        root.after(1000, predict_new_picture)

    label1 = tk.Label(root, text='', font=("Helvetica", 32))
    label1.pack(side="bottom")
    label2 = tk.Label(root, text='', font=("Helvetica", 32))
    label2.pack(side="bottom")
    bild1 = tk.PhotoImage(file='', master=root)
    label3 = tk.Label(root, image=bild1).pack(side="right")
    predict_new_picture()
    root.mainloop()