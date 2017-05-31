#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <list>

#include "Layer.h"

#define MAX_LAYERS 256
#define MAX_PATTERN_LGT 64

using namespace std;

//structure holding config parameters
struct ConfigData
{
	//learning parameters
	bool bias;
	bool randomize_weights;			//randomize neuron weights
	bool randomize_patterns;
	double learning_step;
	double momentum;
	int epochs;						//maximum number of learning epochs
	double precision;				//satisfying learning error value

	//data paths
	string input_network_path;
	string output_network_path;
	string testing_set_path;

	//logging parameters
	string log_path;
	bool log_voice;					//logging output also to std.out
	int log_step;					//how many epochs are to be logged
	bool sy;						//log output of each layer every log_step
	bool sn;						//log network every log_step
	bool lt;						//log learning details after learning process
	bool ln;						//log networl after learning process
};

//testing pattern structure
struct Pattern
{
	int nIn;
	int nOut;
	double in [MAX_PATTERN_LGT];
	double out [MAX_PATTERN_LGT];
};

//exception class
struct FileNotFound
{
	string filename;
	FileNotFound( const string& filename_ = string() )
		: filename(filename_) {}
};

struct UnexpectedEOF
{
	string filename;
	UnexpectedEOF( const string& filename_ = string() )
		: filename(filename_) {}
};

class Perceptron
{
private:
	//data
	ConfigData* conf;

	int nLayers;
	int epoch;
	Layer* layers [MAX_LAYERS];					//network layers
	Layer* outputLayer;

	ofstream logfile, logfile100;
	stringstream ss;
	char logdata [512];

	//methods
	int LoadNetwork(string filename);
	int LN2(string filename);
	
	int StoreNetwork(string filename);
	int LoadTestingSets(string filename);
	void RandomizeWeights();
	void RandomizePatterns();
	int Learn();
	int Test();
	double LearnEpoch();
	int PropagateForward(Pattern* pattern);
	int PropagateBackward(Pattern* pattern);

	int _100Networks();
	void Backup();
	void Restore();

	//logging
	void LogWrite(string s);
	void LogWrite(stringstream* ss);
	void LogInitial();
	void LogEpoch(double xi);
	void LogPattern(Pattern* pattern, int n);
	void LogNetwork();
	void LogFinal();

public:
	list<Pattern> TestingSets;
	int nPatterns;

	Perceptron (ConfigData* cfg);
	~Perceptron();
	int Run();									//main routine
	bool GetBias() {return conf->bias;};
	double GetLearningStep() {return conf->learning_step;};
	double GetMomentum() {return conf->momentum;};
	Layer* GetOutputLayer() {return outputLayer;};
	Layer** GetLayers() {return layers;};
};

#endif
