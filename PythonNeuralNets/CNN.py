import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import random
from math import log
import time

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
    def FeedForward(self,prevA):
        self.Z = np.dot(self.Weights,prevA) + self.Biases
        self.A = Sigmoid(self.Z)

class Softmaxlayer:
    def __init__(self, InSize, OutSize):
        self.Weights = np.random.rand(OutSize,InSize)*np.sqrt(2/InSize)
        self.Biases = np.zeros((OutSize,1))
    def FeedForward(self,prevA):
        self.Z = np.dot(self.Weights,prevA) + self.Biases
        self.A = Stable_softmax(self.Z)

class PoolingLayer:
    def __init__(self,pooling_shape,input_shape):
        self.pooling_w, self.pooling_h = pooling_shape
        self.channels, self.input_h, self.input_w = input_shape
        if self.input_h%self.pooling_h + self.input_w%self.pooling_w != 0:
            print("check pooling shape")
        self.output = [np.zeros((int(self.input_h/self.pooling_h), int(self.input_w/self.pooling_w)))] * self.channels
        self.coefs = [np.zeros((self.input_h, self.input_w))] * self.channels #channels*inputw*inputh
    def FeedForward(self,Input):
        for c in range(self.channels):
            for i in range(self.output[0].shape[0]): #rows
                for j in range(self.output[0].shape[1]): #columns
                    start_i = i * self.pooling_h
                    end_i = start_i + self.pooling_h
                    start_j = j * self.pooling_w
                    end_j = start_j + self.pooling_w
                    block_max = np.max(Input[c][start_i: end_i, start_j : end_j])
                    self.output[c][i, j] = block_max
                    for k in range(start_i, end_i):
                        for l in range(start_j, end_j):
                            if Input[c][k,l] == block_max:
                                self.coefs[c][k,l] = 1
                                #break -> this mean we arbitrarly keep only one max if there are several equal max values
    def BackProp(self,error):
        output = [np.zeros(self.coefs[0].shape)] * self.channels
        for c in range(self.channels):
            for i in range(self.output[0].shape[0]): #rows
                for j in range(self.output[0].shape[1]): #columns
                    start_i = i * self.pooling_h
                    end_i = start_i + self.pooling_h
                    start_j = j * self.pooling_w
                    end_j = start_j + self.pooling_w
                    for k in range(start_i, end_i):
                        for l in range(start_j, end_j):
                            output[c][k,l] = error[c][i,j]
        return np.multiply(output,self.coefs)

class Conv2D:
    def __init__(self,input_shape,filters_shape,nchannels,padding=0): #filters_shape = nfilters * filter_h * filter_w | ex: 10*5*5 for 10 filters with shape (5,5)
        self.nfilters, self.filter_h, self.filter_w = filters_shape
        self.padding = padding
        input_h, input_w = input_shape
        self.nchannels = nchannels #input channels
        self.filters = []
        self.output_shape = (input_shape[0]-self.filter_h+2*self.padding+1,input_shape[1]-self.filter_w+2*self.padding+1)
        for i in range(self.nfilters):
            filter = [] #those filters are summed together in FF
            for j in range(nchannels): #one different weight matrix for each input channel
                filter.append(np.random.rand(self.filter_h,self.filter_w)*0.3-0.15) #TODO: CHECK STDEV
            self.filters.append(filter)
        #self.output_shape = (input_h + filter_h - 1, input_w + filter_w - 1)
    def FeedForward(self,Input):
        output = []
        for c in range(self.nchannels):
            InputCols = im2col(Input[c],self.filter_h, self.filter_w,self.padding)
            for filter in self.filters:
                kernelsSum = np.zeros((InputCols.shape[1],1))
                for kernel in filter:
                    kernelCols = im2col(kernel,self.filter_h, self.filter_w) #never pad the kernel
                    kernelsSum += np.matmul(InputCols.T,kernelCols)
                kernelsSum = kernelsSum.reshape(self.output_shape)
                output.append(kernelsSum)
        self.FeaturesMap_Z = output
        self.FeaturesMap_A = Sigmoid(output)


def im2col(img, filter_h, filter_w, pad=0):
    img_h, img_w = img.shape
    img = np.pad(img, ((pad,pad),(pad,pad)), 'constant')
    img_h += 2*pad
    img_w += 2*pad
    cols = []
    for i in range(img_h-filter_h+1): #rows
        for j in range(img_w-filter_w+1): #columns
            start_i = i
            end_i = i + filter_h
            start_j = j
            end_j = j + filter_w
            col = (img[start_i: end_i, start_j : end_j]).flatten().reshape(filter_h*filter_w)
            cols.append(col)
    return np.asarray(cols).T

def col2im(cols, filter_h, filter_w, output_shape, pad=0): #useless function (reverse im2col operation)
    col_h, col_w = cols.shape
    out_h, out_w = output_shape
    out_h += 2*pad
    out_w += 2*pad
    output = np.zeros((out_h, out_w))
    blocks = []
    for col in cols.T: #columns
        block = col.reshape(filter_h,filter_w)
        blocks.append(block)
    c = 0
    for i in range(out_h-filter_h+1): #rows
        for j in range(out_w-filter_w+1): #columns
            output[i:i+filter_h, j:j+filter_w] = blocks[c]
            #(output)
            c += 1
    return output

t0 = time.time()
class Training:
    def __init__(self):
        self.lr = 0.001
        self.dp = 0.2
    def train(self,epochs):
        loss = []
        print("Input : (28, 28)")
        Conv2D_1 = Conv2D((28,28),(10,5,5),1,2) #10 (5x5) filters
        print("Conv2D (Sigmoid): " + str(Conv2D_1.output_shape))
        MaxPooling_1 = PoolingLayer((2,2), (10,) + Conv2D_1.output_shape)
        print("MaxPooling: " + str(MaxPooling_1.output[0].shape))
        Conv2D_2 = Conv2D(MaxPooling_1.output[0].shape,(10,3,3),10,1)
        print("Conv2D (Sigmoid):" + str(Conv2D_2.output_shape))
        MaxPooling_2 = PoolingLayer((2,2), (10,) + Conv2D_2.output_shape)
        print("MaxPooling: " + str(MaxPooling_2.output[0].shape))
        #flatten layer here
        print("Flattened: " + str(MaxPooling_2.output[0].shape[0]*MaxPooling_2.output[0].shape[1]*MaxPooling_2.channels))
        Dense_1 = Sigmoidlayer(MaxPooling_2.output[0].shape[0]*MaxPooling_2.output[0].shape[1]*MaxPooling_2.channels,500)
        print("Dense (Sigmoid): 500")
        OutputLayer = Softmaxlayer(500,10)
        print("Dense (SoftMax): 10")
        for i in range(epochs): #train
            input = [np.asarray(X_train[i]).reshape(28,28)]
            label = Y_train[i].reshape(10,1)
            Conv2D_1.FeedForward(input)
            MaxPooling_1.FeedForward(Conv2D_1.FeaturesMap_A)
            Conv2D_2.FeedForward(MaxPooling_1.output)
            MaxPooling_2.FeedForward(Conv2D_2.FeaturesMap_A)
            Flattened = np.asarray(MaxPooling_2.output).flatten().reshape(MaxPooling_2.output[0].shape[0]*MaxPooling_2.output[0].shape[1]*MaxPooling_2.channels,1)
            dp = np.random.binomial(1, self.dp, size=(MaxPooling_2.output[0].shape[0]*MaxPooling_2.output[0].shape[1]*MaxPooling_2.channels,1)) / self.dp #dropout
            Flattened *= dp

            Dense_1.FeedForward(Flattened)
            OutputLayer.FeedForward(Dense_1.A)
            #print(OutputLayer.A)

            #BP error
            loss.append(Cross_entropy(OutputLayer.A,Y_train[i]))
            OutputLayerError = OutputLayer.A - label #output error (cross entropy)
            Dense_1Error = np.multiply(np.dot(OutputLayer.Weights.T,OutputLayerError),dSigmoid(Dense_1.Z)) #hidden layer error (BP)
            MaxPooling_2Error = (np.dot(Dense_1.Weights.T,Dense_1Error)*dp).reshape(MaxPooling_2.channels,MaxPooling_2.output[0].shape[0],MaxPooling_2.output[0].shape[1]) #with dropout
            Conv2D_2Error = MaxPooling_2.BackProp(MaxPooling_2Error) #from now we have errors for each channel separatly
            for i in range(len(Conv2D_2Error)): #BP sigmoid
                Conv2D_2Error[i] = np.multiply(Conv2D_2Error[i],dSigmoid(Conv2D_2.FeaturesMap_Z[i]))
            MaxPooling_1Error = []
            for i in range(len(Conv2D_2.filters)):
                #CHECK PADDING VALUE - Conv2D_2.filter_w-1 as padding for im2col to do full convolution
                ChannelError = np.dot(im2col(Conv2D_2Error[i], Conv2D_2.filter_h, Conv2D_2.filter_w, Conv2D_2.filter_w-Conv2D_2.padding-1).T,im2col(np.sum(Conv2D_2.filters[i], axis=0), Conv2D_2.filter_h, Conv2D_2.filter_w)).reshape(MaxPooling_1.output[0].shape)
                MaxPooling_1Error.append(ChannelError)
            Conv2D_1Error = MaxPooling_1.BackProp(np.stack(MaxPooling_1Error))
            for i in range(len(Conv2D_1Error)): #BP sigmoid
                Conv2D_1Error[i] = np.multiply(Conv2D_1Error[i],dSigmoid(Conv2D_1.FeaturesMap_Z[i]))


            #update weights and biases
            OutputLayer.Weights -= self.lr*np.dot(OutputLayerError,Dense_1.A.T)
            OutputLayer.Biases -= self.lr*OutputLayerError
            Dense_1.Weights -= self.lr*np.dot(Dense_1Error,np.asarray(MaxPooling_2.output).flatten().reshape(MaxPooling_2.output[0].shape[0]*MaxPooling_2.output[0].shape[1]*MaxPooling_2.channels,1).T)
            Dense_1.Biases -= self.lr*Dense_1Error
            for i in range(len(Conv2D_2.filters)): #for each output channel #CHECK THIS
                slices = im2col(MaxPooling_1.output[i], Conv2D_2.filter_h, Conv2D_2.filter_w, Conv2D_2.padding).T
                error = Conv2D_2Error[i].flatten()
                for j in range(Conv2D_2.nchannels): #for each kernel
                    for k in range(len(slices)):
                        WeightUpdate = slices[k].reshape(Conv2D_2.filter_h,Conv2D_2.filter_w)*error[k]
                        Conv2D_2.filters[i][j] -= self.lr*WeightUpdate
            for i in range(Conv2D_1.nchannels):
                slices = im2col(input[i], Conv2D_1.filter_h, Conv2D_1.filter_w, Conv2D_1.padding).T
                error = Conv2D_1Error[i].flatten()
                for j in range(len(Conv2D_1.filters[0])):
                    for k in range(len(slices)):
                        WeightUpdate = slices[k].reshape(Conv2D_1.filter_h,Conv2D_1.filter_w)*error[k]
                        Conv2D_1.filters[i][j] -= self.lr*WeightUpdate




            self.lr *= (1. / (1. + 0.00005)) #pretty good fixed decay 0.0001, with lr 0.65

        Testsuccess = 0
        Testfails = 0
        for i in range(len(X_test)):
            input = [np.asarray(X_test[i]).reshape(28,28)]
            label = Y_train[i].reshape(10,1)
            Conv2D_1.FeedForward(input)
            MaxPooling_1.FeedForward(Conv2D_1.FeaturesMap_A)
            Conv2D_2.FeedForward(MaxPooling_1.output)
            MaxPooling_2.FeedForward(Conv2D_2.FeaturesMap_A)
            Flattened = np.asarray(MaxPooling_2.output).flatten().reshape(MaxPooling_2.output[0].shape[0]*MaxPooling_2.output[0].shape[1]*MaxPooling_2.channels,1)
            Dense_1.FeedForward(Flattened)
            OutputLayer.FeedForward(Dense_1.A)
            #print(OutputLayer.A)
            if np.argmax(OutputLayer.A)==np.argmax(Y_test[i]):
                Testsuccess += 1
            else:
                Testfails += 1

        print(Testsuccess/(Testsuccess + Testfails))
        plt.plot(loss)
        plt.show()

Training().train(1000) #7000 -> 0.87
print(time.time()-t0)
