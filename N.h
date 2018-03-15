#pragma once

#include "Header.h"


class N {
public:

	N(std::initializer_list<int>& t, double LearnRate = 0.9, activationMethodchoosen act_method_received = activationMethodchoosen::eins_durch_ehoch, std::tuple<double, double, double, double> nP = { 1.0, 0.0, 1.0, 0.0 }, randomInit in = { -1.0, 1.0 } );
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
	void   norm  (double& p_v_orig); // Normalization function
	double denorm(double& p_v_norm); // Denormalization function

	void(*p_activationfunction)(double * val);
	double(*p_slope)(double * val);

	double ** nod;
	//std::unique_ptr<double *> nod = nullptr;
	double ** err;
	double *** wij; // std::unique_ptr<double ***> wij = nullptr;
	double * den; // denormalized result of calc, only returned on demand
	int Nlay;
	int Nnod;
	int Nwij;
	double* getCalcRes();




public:
	~N();

};


class U {
public:

	U(std::initializer_list<int>& t, double LearnRate = 0.9, activationMethodchoosen act_method_received = activationMethodchoosen::eins_durch_ehoch, std::tuple<double, double, double, double> nP = { 1.0, 0.0, 1.0, 0.0 });
	//double * input;
	//inpPtr input;

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
	void denorm(double& p_v_norm, double& denormedVal);
	void(*p_activationfunction)(double * val);

public:
	//double ** nod;
	std::unique_ptr<std::unique_ptr<double[]>[]> nod = nullptr;

	//std::unique_ptr<double[]>& input;// = nod[0]; // nullptr;
private:
	double ** err;
	//double *** wij; 
	std::unique_ptr<std::unique_ptr<std::unique_ptr<double[]>[]>[]> wij = nullptr;
	double * den; // denormalized result of calc, only returned on demand
	int Nlay;
	int Nnod;
	int Nwij;
	double* getCalcRes();

public:
	~U();

};


class M {
public:
	M(std::initializer_list<int>& t, double LearnRate = 0.9, 
		activationMethodchoosen act_method_received = activationMethodchoosen::eins_durch_ehoch, 
		std::tuple<double, double, double, double> nP = { 1.0, 0.0, 1.0, 0.0 });

	void M::calc(bool doLearn);

//private:
	std::vector<int> top;
	double LearnRate;
	activationMethodchoosen act_method;
	std::tuple<double, double, double, double> normalizationParam;

	double A_max, A_min, new_A_max, new_A_min;
	void   norm(double& p_v_orig); // Normalization function
	double denorm(double& p_v_norm); // Denormalization function

	void(*p_activationfunction)(double * val);
	double(*p_slope)(double * val);

	int Nlay;
	int Nnod;
	int Nwij;

	double ** A, ** B, ** C; // the three matrices for MKL cblas_dgemm(...)
	int * Na, * Nb, * Nc;
	int m, n, k, i, j;
	double alpha, beta;

	double * trueVal;
	double ** err;

	const int D = 1;


};







