#include "Layer.h"
#include "Neuron.h"
#include "Perceptron.h"
#include <list>

Layer::Layer(int _nNeurons, Layer* prev, Perceptron* _p, int _no) : 
nNeurons(_nNeurons),
PreviousLayer(prev),
p(_p),
no(_no)
{
	//initialize neurons
	for (int i = 0; i < nNeurons; i++)
		neurons [i] = new Neuron (this, prev, i);

}

void Layer::Activate()
{
	//activate all neurons
	for (int n = 0; n < nNeurons; n++)
		neurons[n]->Activate();
}

void Layer::CalcDWs(Pattern* pattern)
{
	//for all neurons
	for (int n = 0; n < nNeurons; n++)
		neurons[n]->CalcDW(pattern);
}

void Layer::ClearDWs()
{
	//for all neurons
	for (int n = 0; n < nNeurons; n++)
		neurons[n]->ClearDW();
}

void Layer::ModifyWeights()
{
	//for all neurons
	for (int n = 0; n < nNeurons; n++)
		neurons[n]->ModifyWeights();
}

Layer::~Layer()
{

}

void Layer::Backup()
{
	for (int i = 0; i < nNeurons; i++)
	{
		neurons[i]->Backup();
	}
}

void Layer::Restore()
{
	for (int i = 0; i < nNeurons; i++)
	{
		neurons[i]->Restore();
	}
}