#include "Neuron.h"
#include "Layer.h"
#include "Perceptron.h"
#include <cmath>

Neuron::Neuron(Layer* own, Layer* prev, int _no) : 
OwnLayer(own),
PreviousLayer(prev),
no(_no)
{
	output = 0;
	weightBias = 0;
}

double Neuron::Activate()
{
	//if first layer then exit
	if (PreviousLayer == NULL) return -1;

	//calculate weighted sum of input
	inputWeightedSum = 0;
	double weightedInput = 0;
	for (int w = 0; w < PreviousLayer->nNeurons; w++)
	{
		weightedInput = PreviousLayer->neurons[w]->output * weights[w];
		inputWeightedSum += weightedInput;
	}

	//use bias
	if (OwnLayer->p->GetBias())
	{
		inputWeightedSum += 1.0f * weightBias;
	}

	//logistic activation function
	//output = 1.0 / (exp(inputWeightedSum * (-1.0)) + 1.0);
	output = Sigmoid(inputWeightedSum);
	return output;
}

double Neuron::CalcDelta(Pattern* pattern)
{
	double outDiff;
	if (OwnLayer == OwnLayer->p->GetOutputLayer())
	{
		//this is an output layer
		outDiff = pattern->out[no] - output;
		delta = (-1.0) * SigmoidDerivative(inputWeightedSum) * outDiff;
	}
	else
	{
		delta = 0;
		//return 0;/////////////////////////////DEBUG

		//this is a hidden layer
		double wdsum = 0;
		Layer* nextLayer = OwnLayer->p->GetLayers()[OwnLayer->no + 1];
		for (int nln = 0; nln < nextLayer->nNeurons; nln++)	//for each neuron in the next layer
		{
			//calculate sum of weight * delta
			wdsum += nextLayer->neurons[nln]->weights[no] * nextLayer->neurons[nln]->delta;
		}
		//multiply by derivative of input weighted sum
		delta = wdsum * SigmoidDerivative(inputWeightedSum);
	}

	return delta;
}

double Neuron::CalcDW(Pattern* pattern)
{
	CalcDelta(pattern);
	for (int n = 0; n < PreviousLayer->nNeurons; n++)	//for each weight
	{
		//get dw from previous pattern
		dw[n] *= OwnLayer->p->GetMomentum();
		dw[n] += OwnLayer->p->GetLearningStep() * delta * PreviousLayer->neurons[n]->output;
	}

	dwBias *= OwnLayer->p->GetMomentum();
	dwBias += OwnLayer->p->GetLearningStep() * delta;

	return 1;
}

void Neuron::ClearDW()
{
	for (int w = 0; w < PreviousLayer->nNeurons; w++)
		dw[w] = 0;

	dwBias = 0;
}

double Neuron::Sigmoid(double a)
{
	double output = 1.0 / (exp(a * (-1.0)) + 1.0);
	return output;
}

double Neuron::SigmoidDerivative(double  a)
{
	//double expA = exp(a);
	//return (expA / (expA + 1) * (expA + 1));

	double out = Sigmoid(a) * (1 - Sigmoid(a));
	return out;
}

void Neuron::ModifyWeights()
{
	//for each weight
	for (int w = 0; w < PreviousLayer->nNeurons; w++)
		weights[w] -= dw[w];

	weightBias -= dwBias;
}

void Neuron::Backup()
{
	for (int i = 0; i < PreviousLayer->nNeurons; i++)
	{
		weights2[i] = weights[i];
	}
}

void Neuron::Restore()
{
	for (int i = 0; i < PreviousLayer->nNeurons; i++)
	{
		weights[i] = weights2[i];
	}
}