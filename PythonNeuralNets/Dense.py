import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import random
from math import log
import copy

mnist = input_data.read_data_sets("mnist/", one_hot=True)

X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
Y_train = mnist.train.labels

X_test = np.vstack([img.reshape(-1,) for img in mnist.test.images])
Y_test = mnist.test.labels

def sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)

def dsigmoid(x):
    sig = sigmoid(x)
    return sig*(1-sig)

def Stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

def Cross_entropy(prediction, target): #FIX THIS
    ce = -np.sum(target*np.log(prediction+1e-20))/10.0
    return ce

Sigmoid = np.vectorize(sigmoid)
dSigmoid = np.vectorize(dsigmoid)

class Sigmoidlayer:
    def __init__(self, InSize, OutSize):
        self.Weights = np.random.rand(OutSize,InSize)*np.sqrt(2/InSize)
        self.Biases = np.zeros((OutSize,1))
    def feedforward(self,prevA):
        self.Z = np.dot(self.Weights,prevA) + self.Biases
        self.A = Sigmoid(self.Z)

class Softmaxlayer:
    def __init__(self, InSize, OutSize):
        self.Weights = np.random.rand(OutSize,InSize)*np.sqrt(2/InSize)
        self.Biases = np.zeros((OutSize,1))
    def feedforward(self,prevA):
        self.Z = np.dot(self.Weights,prevA) + self.Biases
        self.A = Stable_softmax(self.Z)

class Training:
    def __init__(self):
        self.lr = 0.65
        self.prevTestVal = 0
        self.p = 0.5 #dropout probability
        self.plateaudecay = True
        self.dropout = False
    def train(self,epochs):
        loss = []
        hl1 = Sigmoidlayer(28*28,800)
        l2 = Softmaxlayer(800,10)
        for i in range(epochs): #train

            dp = np.random.binomial(1, self.p, size=(800,1)) / self.p #2 layer network is too small for dropout

            input = X_train[i].reshape(28*28,1)
            label = Y_train[i].reshape(10,1)
            hl1.feedforward(input)
            if self.dropout:
                hl1.A *= dp
            l2.feedforward(hl1.A)

            #BP error
            loss.append(Cross_entropy(l2.A,Y_train[i]))
            l2error = l2.A - label #output error
            hl1error = np.multiply(np.dot(l2.Weights.T,l2error),dSigmoid(hl1.Z)) #hidden layer error (BP)
            if self.dropout:
                hl1error *= dp

            #update weights and biases
            l2.Weights -= self.lr*np.dot(l2error,hl1.A.T)
            l2.Biases -= self.lr*l2error
            hl1.Weights -= self.lr*np.dot(hl1error,input.T)
            hl1.Biases -= self.lr*hl1error


            #elif epochs>5000:
            #    lr = 0.05
            #elif epochs>2500:
            #    lr = 0.1
            #elif epochs>1000:
            #    lr = 0.05

            #if i == 10000:
            #    self.lr /= 0.9

            self.lr *= (1. / (1. + 0.00005)) #pretty good fixed decay 0.0001, with lr 0.65

            #decay 0.0000126 0.63/50000 -> 0.8 lr 0.63
            # 0.00002 -> 0.87
            # 0.00004 -> 0.9394
            # 0.00005 lr 0.65 -> 0.94

            #self.lr = (self.lr / (1. + (0.63/50000)*i))

            #if i % 10000 == 0: -> lr = 0.63 : val 0.65
            #    self.lr *= 0.05

            if self.plateaudecay and i > 30000 and i % 1000 == 0: #reduce lr on test validation plateau #
                #print("iter " + str(i) + " loss : " + str(Cross_entropy(l2.A,Y_train[i])))
                Testsuccess = 0
                Testfails = 0
                for j in range(len(X_test)//3):
                    hl1.feedforward(X_test[j].reshape(28*28,1))
                    l2.feedforward(hl1.A)
                    if np.argmax(l2.A)==np.argmax(Y_test[j]):
                        Testsuccess += 1
                    else:
                        Testfails += 1
                TestVal = Testsuccess/(Testsuccess + Testfails)
                print("test validation on iteration " + str(i) + " is " + str(TestVal))
                if ((self.prevTestVal!=0) and ((TestVal-self.prevTestVal)/self.prevTestVal)<0.002): #plateau at 0.01 variation or less between 5000 epochs
                    print("plateau... " + str((TestVal-self.prevTestVal)/self.prevTestVal))
                    self.lr *= 0.4
                self.prevTestVal = TestVal


        Testsuccess = 0
        Testfails = 0

        for i in range(len(X_test)):
                hl1.feedforward(X_test[i].reshape(28*28,1))
                l2.feedforward(hl1.A)
                if np.argmax(l2.A)==np.argmax(Y_test[i]):
                    Testsuccess += 1
                else:
                    Testfails += 1

        print(Testsuccess/(Testsuccess + Testfails))
        plt.plot(loss)
        plt.show()

Training().train(50000) #7000 -> 0.87
