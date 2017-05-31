#ifndef NEURON_H
#define NEURON_H

#include <cstddef>

class Layer;
struct Pattern;

#define MAX_NEURONS 256

class Neuron
{
public:
	Layer* OwnLayer;
	Layer* PreviousLayer;
	int no;										//nuber of the neuron within layer
	double weights [MAX_NEURONS];				//weights list against neurons of a previous layer
	double weights2 [MAX_NEURONS];
	double dw [MAX_NEURONS];					//weight modifications during back propagation
	double weightBias;
	double dwBias;

	double output;								//neuron data value processed by activation function
	double inputWeightedSum;
	double delta;

	Neuron(Layer* own, Layer* prev, int _no);	//default constructor
	double Activate();							//calculate weighted input sum and activation function value
	void ClearDW();								//clears dw vector
	double CalcDW(Pattern* pattern);			//calculate weight modification
	double CalcDelta(Pattern* pattern);
	double Sigmoid(double a);
	double SigmoidDerivative(double a);		//derivative of an activation function
	void ModifyWeights();						//applies all dws modifying weights

	void Backup();
	void Restore();
};

#endif
