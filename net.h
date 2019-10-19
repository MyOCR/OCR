#ifndef NET_H
#define NET_H

#include "matrix.h"

typedef struct
{
	size_t nLayer;
	matrix **weights;
	matrix **biases;
}network;

typedef struct
{
	matrix *input;
	matrix *output;
}data;

network *newNet(size_t numLayer, size_t numNodeForLayer[]);
void freeNet(network *net);

double sigmoid(double x);
double sigmoid_prime(double x);
double randInit(double x);

matrix *feedforwardNet(network *net, matrix *input);

void SGD(network *net, data *training_data, size_t training_data_size, 
		size_t epochs, size_t mini_batch_size, double eta , 
		data *test_data, size_t test_data_size);
void updateMiniBatchNet(network *net, data *mini_batch, size_t mini_batch_size, double eta);
void backpropNet(network *net, matrix **nabla_b, matrix **nabla_w, data d);

size_t evaluateNet(network *net, data *test_data, size_t test_data_size);

void printNet(network *net);

#endif //NET_H
