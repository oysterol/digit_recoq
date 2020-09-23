import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

####Variables, check report for how they are deduced
def Z(W,a_ml): 
    return a_ml.dot(W)

def d_k(a_k,y, z): 
    return np.multiply((np.subtract(a_k,y)),dsigma(z))

def d_l(d_m1, z, w_l):
    return np.multiply((d_m1.dot(w_l.T)),dsigma(z))


def sigma(z): 
    for i in range(len(z)):
        z[i]=  1/(1+np.exp(-z[i]))
    return z
def dsigma(z): 
    return np.divide(np.exp(-z),np.multiply((1+np.exp(-z)),(1+np.exp(-z))))

def L_W(d_m, a_ml):
    return (a_ml).T.dot(d_m)
    
def dW(W,L_w,alpha):
    return np.subtract(W,alpha*L_w)

def dB(d_l,B,alpha):
    return np.subtract(B,d_l)

def create_weight(inn, nodes):
    return np.random.rand(inn,nodes)

def create_bias(nodes):
    return np.random.rand(1,nodes)

def define_Neural_network(inn,layers, out):
    weights=list()
    bias=list()

    for i in range(len(layers)):
            m=layers[i]
            weights.append(create_weight(inn,m))
            bias.append(create_bias(m))
            inn=m
    weights.append(create_weight(inn,out))
    bias.append(create_bias(out))
    for i in range(len(weights)):
        weights[i]=np.asmatrix(weights[i])
        bias[i]=np.asmatrix(bias[i])
        
    return weights , bias

def Neural_Network_nodes(inn,NNW,bias):
    nodes=[]
    nodes.append(inn)
    for i in range(0,len(NNW)):
        nodes.append((nodes[i].dot(NNW[i]))+bias[i])
    for i in range(len(nodes)):
        nodes[i]=np.asmatrix(nodes[i])
    return nodes


def train_NN(inn_k,out_k,NNW, bias, alpha):
    z=Neural_Network_nodes(inn_k,NNW,bias) ###aquire all the nodes
    a=sigma(z)
    dtemp=0
    deltabias=list()
    deltaNNW=list()
    for i in range(1,len(NNW)+1):
        ##check if its the last element, and calculate error of layer
        if i==1:
            #print(a[-1].size,z[-1].size,Z(NNW[-i],a[-i-1].size))
            d_m=d_k(a[-i],out_k,z[-1])#Z(NNW[-i],a[-i-1]))
    
        else:
            d_m=d_l(dtemp,z[-i],NNW[-i+1])#Z(NNW[-i],a[-i-1])
            
        ###Update Bias
        #bias[-i]=dB(d_m,bias[-i], alpha)
        deltabias.append(dB(d_m,bias[-i], alpha))
         ###The loss function gradient with respect to the weights
        L_w=np.asmatrix(L_W(d_m,a[-i-1]))
        ##update the weights in a layer
        #NNW[-i]=dW(NNW[-i],L_w,alpha)
        deltaNNW.append(dW(NNW[-i],L_w,alpha))
        ##save the error of last layer to use in the next
        dtemp=d_m
    return deltaNNW[::-1], deltabias[::-1]
 