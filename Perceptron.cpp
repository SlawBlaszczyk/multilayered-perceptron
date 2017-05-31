#include "Perceptron.h"
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <vector>

Perceptron::Perceptron(ConfigData *cfg) : conf(cfg)
{
	nLayers = 0;
	nPatterns = 0;
	for (int i = 0; i < MAX_LAYERS; i++)
		layers[i] = NULL;
}

Perceptron::~Perceptron()
{
	for (int i = 0; i < MAX_LAYERS; i++)
		if (layers[i] != NULL)
			delete layers[i];
		else
			break;
}

int Perceptron::LoadNetwork(std::string filename)
{
	cout << "Loading network file " << filename << ".\n";

	ifstream netfile(filename.c_str());
	if (!netfile) throw FileNotFound(filename);

	string line;
	//read network topology
	getline (netfile, line);
	while (line != "[Topology]")
	{
		if (netfile.eof()) throw UnexpectedEOF(filename);
		getline (netfile, line);
	}

	//get topology data
	getline (netfile, line);
	string delim = ";";
	string::size_type delimPos;

	string val_s;		//data value as a string
	double val_d;		//data value as double
	int val_i;			//data value as int
	Layer* LastLayer = NULL;

	delimPos = line.find(delim);
	while (delimPos < string::npos && nLayers < MAX_LAYERS)
	{
		//extract value
		val_s = line.substr(0, delimPos);
		line.replace(0, delimPos + 2, "");
		val_i = atoi (val_s.c_str());

		//add new network layer
		layers[nLayers] = new Layer (val_i, LastLayer, this, nLayers);
		LastLayer = layers[nLayers];
		nLayers ++;

		delimPos = line.find(delim);  //find next space
	}

	if  (!nLayers) throw (int)1;		//no layers to create

	outputLayer = LastLayer;

	//read neuron weight values
	for (int ln = 0; ln < nLayers; ln++)	//for each layer
	{
		//seek for weights section
		while (line != "[Layer]")
		{
			if (netfile.eof()) throw UnexpectedEOF(filename);
			//if (netfile.eof()) break;
			getline (netfile, line);
		}
		for (int nn = 0; nn < layers[ln]->nNeurons; nn++)	//for each neuron
		{
			//read weights
			getline (netfile, line);
			if (line == "") break;
			delimPos = line.find(delim);

			//extract bias
			val_s = line.substr(0, delimPos);
			line.replace(0, delimPos + 2, "");
			val_d = atof (val_s.c_str());
			layers[ln]->neurons[nn]->weightBias = val_d;
			delimPos = line.find(delim);

			int WeightsRead = 0;
			while (delimPos < string::npos)	//for each value
			{
				//extract value
				val_s = line.substr(0, delimPos);
				line.replace(0, delimPos + 2, "");
				val_d = atof (val_s.c_str());

				//store weight value in the neuron
				layers[ln]->neurons[nn]->weights[WeightsRead] = val_d;

				//cout << layers[ln]->neurons[nn]->weights[0] << "\n";

				WeightsRead++;
				delimPos = line.find(delim);  //find next space
			}
		}
	}
	netfile.close();

	return 1;
}

int Perceptron::LN2(string filename)
{

	ifstream netfile(filename.c_str());
	if (!netfile) throw FileNotFound(filename);

	string line;
	string delim = ";";
	string::size_type delimPos;
	string val_s;		//data value as a string
	double val_d;		//data value as double
	int val_i;			//data value as int
	Layer* LastLayer = NULL;

	if  (!nLayers) throw (int)1;		//no layers to create

	outputLayer = LastLayer;

	//read neuron weight values
	for (int ln = 0; ln < nLayers; ln++)	//for each layer
	{
		//seek for weights section
		while (line != "[Layer]")
		{
			if (netfile.eof()) throw UnexpectedEOF(filename);
			//if (netfile.eof()) break;
			getline (netfile, line);
		}
		for (int nn = 0; nn < layers[ln]->nNeurons; nn++)	//for each neuron
		{
			//read weights
			getline (netfile, line);
			if (line == "") break;
			delimPos = line.find(delim);

			//extract bias
			val_s = line.substr(0, delimPos);
			line.replace(0, delimPos + 2, "");
			val_d = atof (val_s.c_str());
			layers[ln]->neurons[nn]->weightBias = val_d;
			delimPos = line.find(delim);

			int WeightsRead = 0;
			while (delimPos < string::npos)	//for each value
			{
				//extract value
				val_s = line.substr(0, delimPos);
				line.replace(0, delimPos + 2, "");
				val_d = atof (val_s.c_str());

				//store weight value in the neuron
				layers[ln]->neurons[nn]->weights[WeightsRead] = val_d;

				//cout << layers[ln]->neurons[nn]->weights[0] << "\n";

				WeightsRead++;
				delimPos = line.find(delim);  //find next space
			}
		}
	}
	netfile.close();

	return 1;
}

int Perceptron::StoreNetwork(std::string filename)
{
	cout << "Saving network file " << filename << ".\n";
	ofstream netfile(filename.c_str());

	netfile << "# " << filename << "\n";
	netfile << "# Multilayer perceptron network structure file\n\n";

	netfile << "[Topology]\n";

	//save topology data
	for (int i = 0; i < nLayers; i++)
		netfile << layers[i]->nNeurons << "; ";

	netfile << "\n" << "\n";

	//save layers data
	for (int i = 0; i < nLayers; i++)		//for each layer
	{
		netfile << "[Layer]\n";

		if (layers[i]->PreviousLayer != NULL)
			for (int j = 0; j < layers[i]->nNeurons; j++)	//for each neuron
			{
				netfile << layers[i]->neurons[j]->weightBias << "; ";
				for (int k = 0; k < layers[i]->PreviousLayer->nNeurons; k++)	//for each weight
				{
					netfile << layers[i]->neurons[j]->weights[k] << "; ";
				}
				netfile << "\n";
			}

		netfile << "\n";
	}

	netfile.flush();
	netfile.close();

	return 1;
}

int Perceptron::LoadTestingSets(std::string filename)
{
	cout << "Loading pattern file " << filename << ".\n";

	ifstream setfile(filename.c_str());
	if (!setfile) throw FileNotFound(filename);

	string line;
	string in;
	string out;
	string val_s;
	double val_d;

	string iodelim = ";";
	string valdelim = ",";
	string::size_type delimPos, dpin, dpout;

	Pattern p;

	getline(setfile, line);
	while (line != "")	//for each pattern
	{

		//extract input and output parts
		delimPos = line.find(iodelim);
		in = line.substr(0, delimPos);
		out = line.erase(0, delimPos+1);

		//extract each value and write into data structure
		dpin = in.find(valdelim);
		dpout = out.find(valdelim);
		int nin = 0, nout = 0;	//amount of values stored

		while (dpin < string::npos)
		{
			val_s = in.substr(0, dpin);
			in.erase(0, dpin+1);
			val_d = atof(val_s.c_str());
			p.in[nin] = val_d;

			nin++;
			p.nIn = nin;
			dpin = in.find(valdelim);
		}
		while (dpout < string::npos)
		{
			val_s = out.substr(0, dpout);
			out.erase(0, dpout+1);
			val_d = atof(val_s.c_str());
			p.out[nout] = val_d;

			nout++;
			p.nOut = nout;
			dpout = out.find(valdelim);
		}

		//add new pattern
		TestingSets.push_back(p);
		nPatterns++;

		getline (setfile, line);	//next line
	}

	//verify pattern length
	if (layers[0]->nNeurons > TestingSets.begin()->nIn || outputLayer->nNeurons > TestingSets.begin()->nOut)
		throw (int) 2;

	setfile.close();

	return 1;
}

void Perceptron::RandomizeWeights()
{
	srand((unsigned int)time(NULL));
	double weightVal;

	if(!conf->randomize_weights) return;

	for (int l = 1; l < nLayers; l++)
		for (int n = 0; n < layers[l]->nNeurons; n++)
		{
			for (int w = 0; w < layers[l-1]->nNeurons; w++)
			{
				weightVal = rand() % 2000;
				weightVal /= 1000;
				weightVal -= 1;
				layers[l]->neurons[n]->weights[w] = weightVal;
			}
			weightVal = rand() % 2000;
			weightVal /= 1000;
			weightVal -= 1;
			layers[l]->neurons[n]->weightBias = weightVal;
		}
}

void Perceptron::RandomizePatterns()
{
	if(!conf->randomize_patterns) return;

	list<Pattern> p2;	//auxiliary patterns list
	list<Pattern>::iterator it;

	//randomize patterns and move to the second list
	for (int pLeft = nPatterns; pLeft > 0; pLeft--)
	{
		int chosen = rand() % pLeft;
		
		//move iterator to the chosen position
		it = TestingSets.begin();
		for (int j=0; j<chosen; j++)
			it ++;

		p2.splice(p2.begin(), TestingSets, it);
	}

	TestingSets.swap(p2);
}

int Perceptron::Learn()
{
	cout << "Learning...\n";

	//randomize weights if desired
	//RandomizeWeights();

	LogInitial();

	epoch = 0;
	double xi = 0;	//error value
	bool terminate = false;

	//clear weight delta vectors
	for (int l = 1; l < nLayers; l++)
		layers[l]->ClearDWs();

	while (!terminate)
	{
		//run learning epoch
		xi = LearnEpoch();

		//check stop conditions
		if (epoch >= conf->epochs)
		{
			terminate = true;
			ss << "Learning process terminated due to maximum number of epochs reached\n"
				<< "\tat epoch " << epoch << " and error value " << xi << ".\n";
			cout << "Learning process terminated due to maximum number of epochs reached\n"
				<< "at epoch " << epoch << " and error value " << xi << ".\n";
			LogWrite(&ss);
		}
		if (xi < conf->precision)
		{
			terminate = true;
			ss << "Learning process terminated due to desired precision reached\n"
				<< "\tat epoch " << epoch << " and error value " << xi << ".\n";
			cout << "Learning process terminated due to desired precision reached\n"
				<< "at epoch " << epoch << " and error value " << xi << ".\n";
			LogWrite(&ss);
		}

		epoch++;
	}

	return epoch;
}

double Perceptron::LearnEpoch()
{
	double patternXi = 0;	//error value for current pattern
	double globalXi = 0;	//global error value
	int patNo = 1;			//pattern number

	if (epoch % conf->log_step == 0)
	{
		ss << "Initiating epoch " << epoch << ".\n\n";
		LogWrite(&ss);
	}

	RandomizePatterns();

	//for each test pattern
	for (list<Pattern>::iterator it = TestingSets.begin(); it != TestingSets.end(); it++)
	{
		Pattern* pattern = &(*it);

		PropagateForward(pattern);
		PropagateBackward(pattern);

		//check pattern error value
		for (int n = 0; n < outputLayer->nNeurons; n++)
			patternXi = pow ((it->out[n] - outputLayer->neurons[n]->output) , 2);

		//add to global epoch error value
		globalXi += patternXi;

		int k = epoch % conf->log_step;
		if (k == 0 && conf->sy) LogPattern(pattern, patNo);
		patNo++;
	}

	LogEpoch(globalXi);

	return globalXi;
}

int Perceptron::PropagateForward(Pattern* pattern)
{
	//load input data from pattern
	for (int n = 0; n < layers[0]->nNeurons; n++)	//for each neuron in the input layer
		layers[0]->neurons[n]->output = pattern->in[n];

	//for each layer except input layer
	for (int l = 1; l < nLayers; l++)
	{
		layers[l]->Activate();
	}

	return 1;
}

int Perceptron::PropagateBackward(Pattern* pattern)
{
	//for each layer except input layer, backwards
	for (int l = nLayers - 1; l > 0; l--)
	{
		//calculate modification values
		layers[l]->CalcDWs(pattern);
	}

	//modify weights
	for (int l = 1; l < nLayers; l++)
		layers[l]->ModifyWeights();

	return 1;
}

int Perceptron::Test()
{
	double globalXi = 0;
	double patternXi = 0;
	int patNo = 1;

	cout << "Testing...\n";

	LogWrite("\nInitiating testing process...\n\n");

	//for each test pattern
	for (list<Pattern>::iterator it = TestingSets.begin(); it != TestingSets.end(); it++)
	{
		Pattern* pattern = &(*it);
		PropagateForward(pattern);

		//check pattern error value
		for (int n = 0; n < outputLayer->nNeurons; n++)
			patternXi = pow ((it->out[n] - outputLayer->neurons[n]->output) , 2);

		//add to global epoch error value
		globalXi += patternXi;

		if (conf->lt) LogPattern(pattern, patNo);
		patNo++;
	}

	LogFinal();

	cout << "Network processing complete. Testing error " << globalXi << ".\n";

	logfile << "Testing complete with global error value " << globalXi << ".\n";

	return 1;
}

void Perceptron::LogEpoch(double xi)
{
	if (epoch % conf->log_step == 0)
	{
		if (conf->sn)
			LogNetwork();

		ss << "Epoch " << epoch << " summary.\n";
		LogWrite(&ss);
		if(conf->sn)	//log network description
		{
		}

		//global error value
		ss << "Global error value: "  << xi << ".\n\n";
		LogWrite(&ss);
	}
}

void Perceptron::LogPattern(Pattern* pattern, int n)
{
	//log pattern data
	ss << "Pattern " << n << ": ((";
	for (int i = 0; i < pattern->nIn; i++)
		ss << pattern->in[i] << ",";
	ss << ");(";
	for (int i = 0; i < pattern->nOut; i++)
		ss << pattern->out[i] << ",";
	ss << "))\n";
	LogWrite(&ss);
	
	//log neuron output values
	for (int l = 0; l < nLayers; l++)
	{
		ss << "Layer " << l << " output: ";
		for (int n = 0; n < layers[l]->nNeurons; n++)
		{
			ss << layers[l]->neurons[n]->output << "; ";
		}
		ss << "\n";
		LogWrite(&ss);
	}
	LogWrite("\n");
}

void Perceptron::LogNetwork()
{
	logfile << "Network structure:\n\n";
	logfile << "[Topology]\n";

	//save topology data
	for (int i = 0; i < nLayers; i++)
		logfile << layers[i]->nNeurons << "; ";

	logfile << "\n" << "\n";

	//save layers data
	for (int i = 0; i < nLayers; i++)		//for each layer
	{
		logfile << "[Layer]\n";

		if (layers[i]->PreviousLayer != NULL)
			for (int j = 0; j < layers[i]->nNeurons; j++)	//for each neuron
			{
				logfile << layers[i]->neurons[j]->weightBias << "; ";
				for (int k = 0; k < layers[i]->PreviousLayer->nNeurons; k++)	//for each weight
				{
					logfile << layers[i]->neurons[j]->weights[k] << "; ";
				}
				logfile << "\n";
			}

		logfile << "\n";
	}
}

void Perceptron::LogInitial()
{
	ss << "Initiating multilayer perceptron network learning procedure.\n";
	ss << "\nInitial configuration:\n\n";
	LogWrite(&ss);

	ss << "Bias: ";
	if(conf->bias) ss << "set\n";
	else ss << "not set\n";

	ss << "Randomize weights: ";
	if(conf->randomize_weights) ss << "set\n";
	else ss << "not set\n";

	ss << "Randomize patterns: ";
	if(conf->randomize_patterns) ss << "set\n";
	else ss << "not set\n";

	ss << "Learning step: " << conf->learning_step << "\n";
	ss << "Momentum: " << conf->momentum << "\n";
	ss << "Epochs: " << conf->epochs << "\n";
	ss << "Precision: " << conf->precision << "\n";
	LogWrite(&ss);

	ss << "Input network path: " << conf->input_network_path << "\n";
	ss << "Output network path: " << conf->output_network_path << "\n";
	ss << "Testing sets path: " << conf->testing_set_path << "\n";
	ss << "Log path: " << conf->log_path << "\n";
	LogWrite(&ss);

	ss << "Log voice: ";
	if(conf->log_voice) ss << "set\n";
	else ss << "not set\n";

	ss << "Log step: " << conf->log_step << "\n";

	ss << "Log output each log step: ";
	if(conf->sy) ss << "set\n";
	else ss << "not set\n";

	ss << "Log network each log step: ";
	if(conf->sn) ss << "set\n";
	else ss << "not set\n";

	ss << "Log testing output: ";
	if(conf->lt) ss << "set\n";
	else ss << "not set\n";

	ss << "Log testing network: ";
	if(conf->ln) ss << "set\n";
	else ss << "not set\n";
	ss << "\n" << "\n";
	LogWrite(&ss);

	LogNetwork();
}

void Perceptron::LogFinal()
{
	if (conf->ln) LogNetwork();

}

void Perceptron::LogWrite(string s)
{
	logfile << s;
	if (conf->log_voice)
		cout << s;
}

void Perceptron::LogWrite(stringstream* ss)
{
	ss->get(logdata, 1024, EOF);
	logfile << logdata;
	logfile.flush();
	if (conf->log_voice)
		cout << logdata;
	ss->clear();
}

void Perceptron::Backup()
{
	for (int i = 1; i < nLayers; i++)
	{
		layers[i]->Backup();
	}
}

void Perceptron::Restore()
{
	for (int i = 1; i < nLayers; i++)
	{
		layers[i]->Restore();
	}
}

int Perceptron::_100Networks()
{
		//open log file
		logfile.open(conf->log_path.c_str());
		
		///100net
		logfile100.open("100.txt");
		conf->precision = 0.001;
		int eps = 0;

		//create network
		LoadNetwork(conf->input_network_path);
		
		// for (int i = 0; i < MAX_LAYERS; i++)
		// if (layers[i] != NULL)
			// delete layers[i];
			
		LoadTestingSets(conf->testing_set_path);

		//for each network
		for(int i=0; i<100; i++)
		{
			cout << "Learning network " << i << " ...\n";
			RandomizeWeights();
			//StoreNetwork("100.net");
			Backup();
			
			//param set 1
			//LN2("100.net");
			//LoadNetwork("input.net");
			Restore();
			conf->learning_step = 0.2;
			conf->momentum = 0.0;
			eps = Learn();
			logfile100 << eps << " ";
			
			//param set 2
			//LN2("100.net");
			Restore();
			conf->learning_step = 0.6;
			conf->momentum = 0.0;
			eps = Learn();
			logfile100 << eps << " ";
			
			//param set 3
			//LN2("100.net");
			Restore();
			conf->learning_step = 0.2;
			conf->momentum = 0.9;
			eps = Learn();
			logfile100 << eps << " ";
			
			//param set 4
			//LN2("100.net");
			Restore();
			conf->learning_step = 0.6;
			conf->momentum = 0.9;
			eps = Learn();
			logfile100 << eps << "\n";
		
		}
		
		// RandomizeWeights();

		// Learn();
		// Test();

		// StoreNetwork(conf->output_network_path);
		
		logfile100.close();
		logfile.close();

	return 1;
}

int Perceptron::Run()
{
	try
	{
		_100Networks();
		
		// //open log file
		// logfile.open(conf->log_path.c_str());

		// //create network
		// LoadNetwork(conf->input_network_path);
		// LoadTestingSets(conf->testing_set_path);

		// RandomizeWeights();

		// Learn();
		// Test();

		// StoreNetwork(conf->output_network_path);
		// logfile.close();
	}
	catch (FileNotFound e)
	{
		cout << "Error: file " << e.filename << " not found.\n";
		return 0;
	}
	catch (UnexpectedEOF e)
	{
		cout << "Error: unexpected end of file in file " << e.filename << ".\n";
		return 0;
	}
	catch (int e)
	{
		cout << "Error " << e << ".\n";
		return 0;
	}

	return 1;
}
