from random import randint
import numpy as np
from final_neural_network import Network
import matplotlib.pyplot as plot
import tkinter as tk
from matplotlib import pyplot as plt

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

networkname = "testnet" #input("Please Enter the Network you want to use to see Working:")

nn = Network([784, 30, 10, 10])


#nn.saveNetwork("testnet")
nn.loadNetwork(networkname)
nn.stochastic_gradient_descent(training_images, training_labels, test_images, test_labels, 10, 10, 3.0)
nn.saveNetwork(networkname)

#print(nn.testdatapackage(test_images, test_labels))

"""root = tk.Tk()

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
root.mainloop()"""