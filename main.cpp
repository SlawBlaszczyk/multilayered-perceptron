#include "Perceptron.h"
#include "ConfigFile.h"

#ifdef WIN32
#include "win32\\getopt.h"
#else
#include <getopt.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#endif



///linux


using namespace std;

int ReadConfig(char* file, ConfigData* cfg)
{
	try
	{
		//opening config file
		cout << "Reading configuration data..." << endl;
		ConfigFile config (file);

		//reading data
		config.readInto(cfg->bias, "bias");
		config.readInto(cfg->randomize_weights, "randomize_weights");
		config.readInto(cfg->randomize_patterns, "randomize_patterns");
		config.readInto(cfg->learning_step, "learning_step");
		config.readInto(cfg->momentum, "momentum");
		config.readInto(cfg->epochs, "epochs");
		config.readInto(cfg->precision, "precision");

		config.readInto(cfg->input_network_path, "input_network_path");
		config.readInto(cfg->output_network_path, "output_network_path");
		config.readInto(cfg->testing_set_path, "testing_set_path");

		config.readInto(cfg->log_path, "log_path");
		config.readInto(cfg->log_voice, "log_voice");
		config.readInto(cfg->log_step, "log_step");
		config.readInto(cfg->sy, "sy");
		config.readInto(cfg->sn, "sn");
		config.readInto(cfg->lt, "lt");
		config.readInto(cfg->ln, "ln");
	}
	catch (ConfigFile::file_not_found e)
	{
		cout << "Error. Config file " << e.filename << " not found. Terminating program." << endl;
		exit(0);
	}

	return 0;
}

void usage()
{
	cout << "\nUsage:\n\n";
	cout << "-c path : set config file path\n";
	cout << "-b : switch on bias\n";
	cout << "-q : randomize weights\n";
	cout << "-r : randomize patterns\n";
	cout << "-l val : set learning step\n";
	cout << "-m val : set momentum\n";
	cout << "-e val : set number of epochs\n";
	cout << "-p val : set desired precision\n";
	cout << "-w path : set input network path\n";
	cout << "-z path : set output network path\n";
	cout << "-d path : set testing sets path\n";
	cout << "-l path : set log path\n";
	cout << "-s val : set log step\n";
	cout << "-v : switch on log voice\n";
	cout << "-a : switch on output logging each log step\n";
	cout << "-i : switch on network logging each log step\n";
	cout << "-f : switch on output logging upon testing\n";
	cout << "-g : switch on network logging upon testing\n";

	return;
}

int main(int argc, char* argv[])
{
	char* version = "1.0";
	char ConfigFileName [256] = "perceptron.conf";
	ConfigData conf = {0};	//structure holding config parameters

	cout << "Perceptron " << version << endl << 
		"by Maciej Zalewski & Slawomir Blaszczyk 2010\n\n";

	//load config file
	ReadConfig(ConfigFileName, &conf);

	//checking command-line parameters
    int c;
     
    opterr = 0;
     
	//TODO: mo�e to usprawni� i doda� error handling
	while ((c = getopt (argc, argv, "c:brqn:m:e:p:w:z:d:l:vhs:aifg")) != -1)
	{
		switch (c)
		{
		case 'c':
#ifndef WIN32
			strncpy(ConfigFileName, optarg,256);  //defining configuration file
#else
			strncpy(ConfigFileName, optarg,256);  //defining configuration file
#endif
			ReadConfig(ConfigFileName, &conf);
			break;
		case 'b':
			conf.bias = true;
			break;
		case 'q':
			conf.randomize_weights = true;
			break;
		case 'r':
			conf.randomize_patterns = true;
			break;
		case 'n':
			conf.learning_step = atof(optarg);
			break;
		case 'm':
			conf.momentum = atof(optarg);
			break;
		case 'e':
			conf.epochs = atoi(optarg);
			break;
		case 'p':
			conf.precision = atof(optarg);
			break;
		case 'w':
			conf.input_network_path = optarg;
			break;
		case 'z':
			conf.output_network_path = optarg;
			break;
		case 'd':
			conf.testing_set_path = optarg;
			break;
		case 'l':
			conf.log_path = optarg;
			break;
		case 'v':
			conf.log_voice = true;
			break;
		case 's':
			conf.log_step = atoi(optarg);
			break;
		case 'a':
			conf.sy = true;
			break;
		case 'i':
			conf.sn = true;
			break;
		case 'f':
			conf.lt = true;
			break;
		case 'g':
			conf.ln = true;
			break;
		case 'h':
			usage();
			return 0;
		case '?':
			if (isprint (optopt))
				fprintf (stderr, "Unknown option `-%c'. -h for usage.\n", optopt);
			else
				fprintf (stderr, "Unknown option character `\\x%x'. -h for usage.\n",optopt);
			return 0;
		default:
			cout << "Creating default network." << endl;
			break;
		}
	}

	//create network instance
	Perceptron* p = new Perceptron(&conf);

	p->Run();

	delete p;

	//system("PAUSE");
	return 0;
}
