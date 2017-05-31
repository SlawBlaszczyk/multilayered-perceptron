#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"

class Perceptron;
struct Pattern;

class Layer
{
public:
	//data
	Layer* PreviousLayer;
	Perceptron* p;
	int no;

	int nNeurons;
	Neuron* neurons [MAX_NEURONS];			//neurons array

	Layer(int _nNeurons, Layer* prev, Perceptron* _p, int _no);		//default constructor
	~Layer();
	void Activate();
	void CalcDWs(Pattern* pattern);
	void ClearDWs();
	void ModifyWeights();

	void Backup();
	void Restore();
};

#endif
