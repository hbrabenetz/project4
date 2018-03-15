#pragma once

#include "Header.h"

class MN {
public:

	MN(std::initializer_list<int>& t, double LearnRate = 0.9, activationMethodchoosen act_method_received = activationMethodchoosen::eins_durch_ehoch, std::tuple<double, double, double, double> nP = { 1.0, 0.0, 1.0, 0.0 }, randomInit in = { -1.0, 1.0 });
	double * input;
	double * trueVal;
	void calc(bool doLearn);
	double * output;

private:
	std::vector<int> top;
	double LearnRate;
	activationMethodchoosen act_method;
	std::tuple<double, double, double, double> normalizationParam;

	double A_max, A_min, new_A_max, new_A_min;
	void   norm(double& p_v_orig); // Normalization function
	double denorm(double& p_v_norm); // Denormalization function

	void(*p_activationfunction)(double * val);
	double(*p_slope)(double * val);

	double ** nod;
	//std::unique_ptr<double *> nod = nullptr;
	double ** err;
	double *** wij; // std::unique_ptr<double ***> wij = nullptr;
	double ** wijMatrix;
	int Nrow;
	int Ncol;
	double * den; // denormalized result of calc, only returned on demand
	int Nlay;
	int Nnod;
	int Nwij;
	int * NwijMatrix;
	double* getCalcRes();

public:
	~MN() {};

};


class MKL {
public:

	MKL(std::initializer_list<int>& t, double LearnRate = 0.9, activationMethodchoosen act_method_received = activationMethodchoosen::eins_durch_ehoch, std::tuple<double, double, double, double> nP = { 1.0, 0.0, 1.0, 0.0 }, randomInit in = { -1.0, 1.0 });
	double * input;
	double * trueVal;
	void calc(bool doLearn);
	double * output;

private:
	std::vector<int> top;
	double LearnRate;
	activationMethodchoosen act_method;
	std::tuple<double, double, double, double> normalizationParam;

	double A_max, A_min, new_A_max, new_A_min;
	void   norm(double& p_v_orig); // Normalization function
	double denorm(double& p_v_norm); // Denormalization function

	void(*p_activationfunction)(double * val);
	double(*p_slope)(double * val);

	double ** nod;
	//std::unique_ptr<double *> nod = nullptr;
	double ** err;
	double *** wij; // std::unique_ptr<double ***> wij = nullptr;
	double ** wijMatrix;
	int Nrow;
	int Ncol;
	double * den; // denormalized result of calc, only returned on demand
	int Nlay;
	int Nnod;
	int Nwij;
	int * NwijMatrix;
	double* getCalcRes();

public:
	~MKL() {};

};
