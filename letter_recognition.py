
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from Neural_network import define_Neural_network
from Letter_recogn_frame import *
import csv

#print('load digits from dataset')
#digits = load_digits()
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

def Create_and_train(dim):
    print('load digits from dataset')
    digits = load_digits()
    # print(digits.data.shape)
    # plt.gray()
    # plt.matshow(digits.images[0])
    # plt.show()
    # digits.target[10]

    # create a Neural Network with random nodes, the layers are created
    # from user input. 64 pixel inputs and 10 different possible outputs
    dimensions = [20,20]
    print('Creating Neural network with dimensions dimensions {}'.format(dimensions))


    W, bias = NN_make(dimensions, 64, 10)
    W, bias = training_montage(100, 0.6, W, bias,digits)
    with open('NN_digits.csv','a') as csvfile:
        writer=csv.writer(csvfile)
        for i in range(len(W)):
            writer.writerow(W[i])


def re_make_NN(NN_file):
    with open(NN_file, 'a') as csvfile:
        values=csv.reader(csvfile, delimiter=',')
        for value in values:

            print(value)


re_make_NN('NN_digits.csv')
#Accuracy(W,bias,0,200,digits)
#print(prediction(W, 211, bias, digits))
#ans=input('save model?? y/n')

