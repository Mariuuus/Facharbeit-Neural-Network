from cgi import test
from random import randint
import numpy as np
from final_neural_network import Network
import final_neural_network as network
import tkinter as tk
from matplotlib import pyplot as plt

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

nn = Network([784, 30, 10, 10])
test_run = False

def trainNetwork():
    print("Was möchtest du machen?")
    print(' - erstelle ein neues Netwzwerk - tippe "create"')
    print(' - traniere ein Netwerk - tippe "train"')
    action = input('"train" or "create":')
    if action == 'train':
        networkname = input('Name des Netzwerkes, dass du trainieren möchtest:')
        if network.doesNetworkExists(networkname):
            nn.loadNetwork(networkname)

            def trainstart():
                nn.stochastic_gradient_descent(training_images, training_labels, test_images, test_labels,int(e1.get()),int(e2.get()),float(e3.get()))
                nn.saveNetwork(networkname)
            root = tk.Tk()
            tk.Label(root, text=('Netzwerk:',networkname), font=("Helvetica", 20), justify='center', pady=10).grid(row=0, column=0)
            tk.Label(root, text="Runden/Epochen").grid(row=1)
            tk.Label(root, text="minibatch-Größe").grid(row=2)
            tk.Label(root, text="lern-Rate").grid(row=3)
            e1 = tk.Entry(root, textvariable=tk.IntVar())
            e2 = tk.Entry(root, textvariable=tk.IntVar())
            e3 = tk.Entry(root, textvariable=tk.DoubleVar())
            e1.grid(row=1, column=1)
            e2.grid(row=2, column=1)
            e3.grid(row=3, column=1)


            tk.Button(root, text="trainieren", font=("Helvetica", 16), command=trainstart).grid(row=4, column=0, pady=25)
            tk.Button(root, text="schließen", font=("Helvetica", 16), command=root.destroy).grid(row=4, column=1, pady=25)

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

#trainNetwork()

def useNetwork():
    networkname = input('Name des Netzwerkes, dass du visuell testen möchtest:')
    nn.loadNetwork(networkname)
    root = tk.Tk()
    
    def predict_new_picture():
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
    network = tk.Label(root, text=('Network: '+networkname), font=("Helvetica", 24))
    network.pack(side='top')
    label1 = tk.Label(root, text='Prediction', font=("Helvetica", 16), justify='left')
    label1.pack(side="bottom")
    label2 = tk.Label(root, text='Right Answer:', font=("Helvetica", 16), justify='left')
    label2.pack(side="bottom")
    bild1 = tk.PhotoImage(file='', master=root)
    tk.Label(root, image=bild1).pack(side="right")
    predict_new_picture()
    root.mainloop()
useNetwork()