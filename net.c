#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <err.h>
#include "net.h"

network *newNet(size_t numLayer, size_t numNodeForLayer[])
{//create a new network with numLayer layers (input layer include)
	if(numLayer < 2)
		errx(1, "Error in newNet. A network has to have at least two layers.");
	for(size_t i = 0; i < numLayer; i++)
	{
		if(numNodeForLayer[i] == 0)
			errx(1, "Error in newNet. A layer has to have at least one node");
	}

	network *net = malloc (sizeof(net));
	net->nLayer = numLayer-1;//the input layer is not counted
	net->weights = malloc (sizeof(matrix*)*net->nLayer);
	net->biases = malloc (sizeof(matrix*)*net->nLayer);

	for(size_t i = 0; i < net->nLayer; i++)
	{
		net->biases[i] = newMat(1, numNodeForLayer[i+1]);
		net->weights[i] = newMat(numNodeForLayer[i], numNodeForLayer[i+1]);
		applyToMat(net->weights[i], randInit);
		applyToMat(net->biases[i], randInit);
	}

	return net;
}

void freeNet(network *net)
{
	for(size_t i = 0; i < net->nLayer; i++)
	{
		freeMat(net->weights[i]);
		freeMat(net->biases[i]);
	}
	
	free(net->weights);
	free(net->biases);
	free(net);
}

double sigmoid(double x)
{
	return 1.0/(1.0+exp(-x));
}

double sigmoid_prime(double x)
{
	double s = sigmoid(x);
	return s*(1-s);
}

double randInit(double x)
{
	return (double)(rand() % 1000)/1000;
}

matrix *feedforwardNet(network *net, matrix *input)
{
	if(input->height != 1 || input->width != net->weights[0]->height)
		errx(1, "Error in feedforwardNet. The input does not have the right size.");

	input = copyMat(input);
	matrix *mat = NULL;
	for(size_t i = 0; i < net->nLayer; i++)
	{
		mat = dotMat(input, net->weights[i]);
		freeMat(input);
		input = addMat(mat, net->biases[i]);
		applyToMat(input, sigmoid);
	}

	return input;
}

void SGD(network *net, data *training_data, size_t training_data_size, 
		size_t epochs, size_t mini_batch_size, double eta , 
		data *test_data, size_t test_data_size)
{
	for(size_t i = 0; i < epochs; i++)
	{
		//TODO shuffle training data
		data **mini_batches = &training_data; //TODO initialize mini_batches correctly
		for(size_t j = 0; j < 1 /*TODO mini_batches_size*/; j++)
		{
			updateMiniBatchNet(net, mini_batches[j], mini_batch_size, eta);
		}

		if(test_data_size)
			printf("Epoch %lu: %lu / %lu.\n", i, evaluateNet(net, test_data, test_data_size), test_data_size);
		else
			printf("Epoch %lu complete.\n", i);
	}
}

void updateMiniBatchNet(network *net, data *mini_batch, size_t mini_batch_size, double eta)
{
	matrix **nabla_b = malloc(sizeof(matrix *)*net->nLayer), **nabla_w = malloc(sizeof(matrix *)*net->nLayer), 
	       **delta_nabla_b = malloc(sizeof(matrix *)*net->nLayer), **delta_nabla_w = malloc(sizeof(matrix *)*net->nLayer);
	double prod = eta/mini_batch_size;

	for(size_t i = 0; i < net->nLayer; i++)
	{
		nabla_b[i] = copySizeMat(net->biases[i]);
		nabla_w[i] = copySizeMat(net->weights[i]);
		//initMat(nabla_b[i], 0);
		//initMat(nabla_w[i], 0);
		//seems useless
	}

	for(size_t i = 0; i < mini_batch_size; i++)
	{
		backpropNet(net, delta_nabla_b, delta_nabla_w, mini_batch[i]);

		for(size_t j = 0; j < net->nLayer; j++)
		{
			addMat(nabla_b[j], delta_nabla_b[j]);
			addMat(nabla_w[j], delta_nabla_w[j]);
			freeMat(delta_nabla_b[j]);
			freeMat(delta_nabla_w[j]);
		}
	}

	for(size_t i = 0; i < net->nLayer; i++)
	{
		subMat(net->weights[i], prodMat(nabla_w[i], prod));
		subMat(net->biases[i], prodMat(nabla_b[i], prod));
	}

	//free all matrices
	for(size_t i = 0; i < net->nLayer; i++)
	{
		freeMat(nabla_b[i]);
		freeMat(nabla_w[i]);
	}

	free(nabla_b);
	free(nabla_w);
	free(delta_nabla_b);
	free(delta_nabla_w);
}

void backpropNet(network *net, matrix **nabla_b, matrix **nabla_w, data d)
{//TODO verifier allocation dynamique
	matrix *activation = d.input, *z = NULL, *delta = NULL, *temp = NULL;
	matrix **activations = malloc(sizeof(matrix *)*(net->nLayer+1));
	activations[0] = copyMat(activation);
	matrix **zs = malloc(sizeof(matrix *)*net->nLayer);

	for(size_t i = 0; i < net->nLayer; i++)
	{
		zs[i] = addMat(dotMat(activation, net->weights[i]), net->biases[i]);

		activation = copyMat(zs[i]);
		activations[i+1] = applyToMat(activation, sigmoid);
	}

	temp = subMat(activations[net->nLayer], d.output);
	delta = prod2Mat(temp, applyToMat(zs[net->nLayer-1], sigmoid_prime));
	freeMat(temp);
	
	nabla_b[net->nLayer-1] = delta;

	activation = transposeMat(activations[net->nLayer-1]);
	nabla_w[net->nLayer-1] = dotMat(activation, delta);
	freeMat(activation);

	for(size_t i = 2; i <= net->nLayer; i++)
	{
		z = zs[net->nLayer-i];
		applyToMat(z, sigmoid_prime);

		temp = transposeMat(net->weights[net->nLayer-i+1]);
		delta = prod2Mat(dotMat(delta, temp), z);

		freeMat(temp);

		nabla_b[net->nLayer-i] = delta;

		temp = transposeMat(activations[net->nLayer-i]);
		nabla_w[net->nLayer-i] = dotMat(temp, delta);
		freeMat(temp);
	}

	for(size_t i = 0; i < net->nLayer; i++)
	{
		freeMat(zs[i]);
		freeMat(activations[i]);
	}
	//is it true? --> do not free activations[net->nLayer]) because it is nabla_b[net->nLayer-1]
	//do not free delta, it is nabla_b[0]
	free(zs);
	free(activations);
}

size_t evaluateNet(network *net, data *test_data, size_t test_data_size)
{
	size_t output = 0;
	matrix *m = NULL;
	for(size_t i = 0; i < test_data_size; i++)
	{
		m = feedforwardNet(net, test_data[i].input);
		//if(test_data[i].output == m)
			//output ++;
			//TODO
	}
	freeMat(m);

	return output;
}

void printNet(network *net)
{
	for(size_t i = 0; i < net->nLayer; i++)
	{
		printMat(net->weights[i]);
		printf("\n\n");
		printMat(net->biases[i]);
		printf("\n\n");
	}
}
