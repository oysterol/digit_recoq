
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from Neural_network import define_Neural_network
from Letter_recogn_frame import *
import csv
import numpy as np

#print('load digits from dataset')
digits = load_digits()
# print(digits.data.shape)
# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()
# digits.target[10]

# create a Neural Network with random nodes, the layers are created
# from user input. 64 pixel inputs and 10 different possible outputs
#dimensions = [20,20]
#print('Creating Neural network with dimensions dimensions {}'.format(dimensions))



def NN_make(dimensions, inn, out):
    return define_Neural_network(inn, dimensions, out)

# Training
def training_montage(r, res, W, bias,digits):
    for i in range(r):
        W, bias = training_NN_digits(W, bias, 0, 1700, res/(i+1)**2, digits)
    return W, bias


#W, bias = NN_make(dimensions, 64, 10)
#W, bias = training_montage(100, 0.6, W, bias)
#f=open('NN_digits.txt','w')

def Create_and_train(dimensions):
    print('load digits from dataset')
    digits = load_digits()


    # create a Neural Network with random nodes, the layers are created
    # from user input. 64 pixel inputs and 10 different possible outputs
    print('Creating Neural network with dimensions dimensions {}'.format(dimensions))


    W, bias = NN_make(dimensions, 64, 10)
    W, bias = training_montage(100, 0.6, W, bias,digits)
    print('Accuracy of model:', Accuracy(W,bias,0,200,digits))
    an=input('Overwrite previous weights? y/n')
    if an=='y':
        file_NN='NN_digits.npy'
        weights_NN=[]
        weights_NN.append([W,bias])
        
        #print(weights_NN)
        np.save(file_NN, weights_NN,allow_pickle=True)
def construct_NN(file):
    Weights=np.load(file, allow_pickle=True)
    W, bias=Weights[0][0], Weights[0][1]
    return W,bias

#Create_and_train([30,30])
W,bias=construct_NN('NN_digits.npy')
#ans=input('save model?? y/n')

