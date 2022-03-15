from random import randint
import numpy as np
from final_neural_network import Network
import final_neural_network as network
import tkinter as tk
from matplotlib import pyplot as plt


"""Diese Datei dient einfach der Darstellung, Visualiesirung und Nutzung eines Neuronalen Netzwerkes"""


#laden der Dateb von MNIST
with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

#erstellen des Netzwerkes
nn = Network([784, 30, 10, 10])

#einloggen in Netzwerke (mit tkinter als Grafikanzeige)
def loadNetwork():
    root = tk.Tk()
    tk.Label(root, text='Log into a network', font=("Helvetica", 24), justify='center', pady=5).grid(row=0)
    tk.Label(root, text='Networkname:', font=("Helvetica", 16)).grid(row=1)
    netname = tk.Entry(root)
    netname.grid(row=2, column=0)

    def logIn():
        tempnetname = netname.get()
        root.destroy()
        if network.doesNetworkExists(tempnetname):
            nn.loadNetwork(tempnetname)
        else:
            nn.saveNetwork(tempnetname)
        configureNetwork(tempnetname)

    tk.Button(root, text="erstellen/konfigurieren", font=("Helvetica", 10), command=logIn).grid(row=3, column=0, pady=5)
    tk.Button(root, text="schließen", font=("Helvetica", 10), command=root.destroy).grid(row=4, column=0, pady=0)
    root.mainloop()

#Netzwerk trainier und Konfigurierfenster, wird nach den erstellen oder einloggen in ein Netzwerk aufegrufen (mit tkinter als Grafikanzeige)
def configureNetwork(networkname):
    def trainstart():
        traingraph = nn.stochastic_gradient_descent(training_images, training_labels, test_images, test_labels,int(e1.get()),int(e2.get()),float(e3.get()))
        nn.saveNetwork(networkname)
        plt.plot(traingraph)
        plt.ylabel('accurency')
        plt.show()
    def testnet():
        root.destroy()
        useNetwork(networkname)
    root = tk.Tk()
    tk.Label(root, text=('Netzwerk:',networkname), font=("Helvetica", 20), justify='center', pady=10).grid(row=0)
    tk.Label(root, text="Runden/Epochen").grid(row=1)
    tk.Label(root, text="minibatch-Größe").grid(row=3)
    tk.Label(root, text="lern-Rate").grid(row=5)
    e1 = tk.Entry(root, textvariable=tk.IntVar())
    e2 = tk.Entry(root, textvariable=tk.IntVar())
    e3 = tk.Entry(root, textvariable=tk.DoubleVar())
    e1.grid(row=2)
    e2.grid(row=4)
    e3.grid(row=6)

    tk.Button(root, text="trainieren", font=("Helvetica", 12), command=trainstart).grid(row=7, pady=5)
    tk.Button(root, text="testen", font=("Helvetica", 12), command=testnet).grid(row=8, pady=5)
    tk.Button(root, text="schließen", font=("Helvetica", 12), command=root.destroy).grid(row=9, pady=5)

    root.mainloop()

#testen eines Netzwerkes, anhand zufälliger Testdaten. (mit tkinter als Grafikanzeige)
def useNetwork(networkname):
    nn.loadNetwork(networkname)
    root = tk.Tk()

    def quit():
        root.after_cancel(after_id)
        root.destroy()

    def back_in_config():
        root.after_cancel(after_id)
        root.destroy()
        configureNetwork(networkname)
    
    def predict_new_picture():
        global after_id
        randompic = randint(0, 10000)
        plt.imshow(training_images[randompic].reshape(28,28), cmap='gray')
        path = './img/'+"current"+'.png'
        plt.savefig(path, dpi=100)
        text1 = "Prediction: "+ str(nn.predictOneInput(training_images[randompic]))
        text2 = "Right Answer: "+ str(np.argmax(training_labels[randompic]))
        label1.config(text=text1)
        label2.config(text=text2)
        bild1.config(file=path)
        after_id = root.after(1000, predict_new_picture)
    network = tk.Label(root, text=('Network: '+networkname), font=("Helvetica", 24))
    network.pack(side='top')
    btn = tk.Button(root, text="schließen", font=("Helvetica", 16), command=back_in_config)
    btn.pack(side="bottom")
    label1 = tk.Label(root, text='Prediction', font=("Helvetica", 16), justify='left')
    label1.pack(side="bottom")
    label2 = tk.Label(root, text='Right Answer:', font=("Helvetica", 16), justify='left')
    label2.pack(side="bottom")
    
    bild1 = tk.PhotoImage(file='', master=root)
    tk.Label(root, image=bild1).pack(side="right")
    predict_new_picture()
    root.protocol('WM_DELETE_WINDOW', quit)
    root.mainloop()

#loadnetwork als anfang des Skriptes.
loadNetwork()
