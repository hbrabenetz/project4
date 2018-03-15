
#include "N.h"
#include <mkl.h>
//#include <mkl_dnn.h> // https://software.intel.com/en-us/mkl-developer-reference-c
//#include <cblas.h>
//#include "mkldnn.hpp"

// Normalization function
void M::norm(double& p_v_orig) { //, double& A_max, double& A_min, double& new_A_max, double& new_A_min) {
	p_v_orig = (p_v_orig - A_min) * (new_A_max - new_A_min) / (A_max - A_min) + new_A_min;
}


// Denormalization function
double M::denorm(double& p_v_norm) { //, double& A_max, double& A_min, double& new_A_max, double& new_A_min) {
	return (p_v_norm - new_A_min) * (A_max - A_min) / (new_A_max - new_A_min) + A_min;
}


// Constructor
M::M(std::initializer_list<int>& topol, double LearnRate, activationMethodchoosen act_method_received, std::tuple<double, double, double, double> normParam) :
	top{ topol }, LearnRate{ LearnRate }, act_method{ act_method_received }, normalizationParam{ normParam }
{

	A_max = get<0>(normalizationParam);
	A_min = get<1>(normalizationParam);
	new_A_max = get<2>(normalizationParam);
	new_A_min = get<3>(normalizationParam);

	if (act_method == activationMethodchoosen::eins_durch_ehoch) {
		p_activationfunction = eins_durch_ehoch;
		p_slope = derivative_eins_durch_ehoch;
	}
	else if (act_method == activationMethodchoosen::ReLU) {
		p_activationfunction = ReLU;
		p_slope = derivative_ReLU;
	}

	Nnod = 0;
	using std::cout;
	cout << "topologie ";
	for (auto e : top) {
		cout << e << " ";
		Nnod += e;
	}
	cout << endl;

	Nlay = static_cast<int> (top.size()); // (int)
	cout << "Nlayer = " << Nlay << endl;

	if (Nlay < 3) {
		cout << "A neural network must have at least three layer including input and output layer. Programm will be terminated." << endl;
		exit(0);
	}

	A = new double* [Nlay - 1]; // for every in between two layers. So 3 layer have 2 matix calculations
	B = new double* [Nlay - 1];
	C = new double* [Nlay - 1];
	Na = new int [Nlay - 1];
	Nb = new int [Nlay - 1];
	Nc = new int [Nlay - 1];

	auto random_d = std::bind(std::uniform_real_distribution<double>(-1.0, 1.0), std::default_random_engine{});

	//const int D = 1; // this is the additional node in each layer for the hybrid D, to avoid an extra D calc

	Nwij = 0;
	for (int nlay = 0; nlay < Nlay - 1; ++nlay) {//Nlay - 1

		//m = 3, k = 2, n = 1;
		k = top[nlay] + D;
		m = top[nlay+1];
		n = 1;

		//cout << "k " << k << " m " << m << "n " << n << endl;

		if(0) printf(" Initializing data for matrix multiplication C=A*B for matrix \n"
					 " A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);

		alpha = 1.0; beta = 0.0;

		if(0) printf(" Allocating memory for matrices aligned on 64-byte boundary for better \n"
					 " performance \n\n");

		//double  randomDebug = 0.0;
		//A[nlay] = (double *)mkl_malloc(m*k * sizeof(double), 64);
		A[nlay] = (double *)malloc(m*k * sizeof(double));
		Nwij += Na[nlay] = m*k;
		for (int na = 0; na < Na[nlay]; ++na) {
			A[nlay][na] = random_d();// randomDebug++; // random_d();
			//cout << "A init " << A[nlay][na] << endl;
		}

		//randomDebug = 0.0;
		//B[nlay] = (double *)mkl_malloc(k*n * sizeof(double), 64);
		B[nlay] = (double *)malloc(k*n * sizeof(double));
		Nb[nlay] = k*n;
		for (int nb = 0; nb < Nb[nlay]; ++nb)
			B[nlay][nb] = 0.0; // randomDebug++; //0.0; // Input values 
		B[nlay][Nb[nlay]-1] = 1.0; // Befor we calc we need to do the D to 1.0

		//randomDebug = 0.0;
		//C[nlay] = (double *)mkl_malloc(m*n * sizeof(double), 64);
		C[nlay] = (double *)malloc((m*n+1) * sizeof(double));
		Nc[nlay] = m*n;
		for (int nc = 0; nc < Nc[nlay]; ++nc)
			C[nlay][nc] = 0.0; //randomDebug++; //0.0; // Input values // here has to be still 0  after calc then to be set at 1
		C[nlay][Nc[nlay]-0] = 1.0;

		// show ABC
		if (0) {

			cout << "nlay == " << nlay << endl;
			cout << "mi*k + i = " << m * k << endl;

			for (int mi = 0; mi < m; ++mi)
				for (int ki = 0; ki < k; ++ki)
					cout << "A[nlay][mi*k + ki] " << "A["<<nlay<<"]["<<mi<<"*"<<k<<" + "<<ki<<"] "<< A[0][mi*k + ki] << endl;


			for (int ki = 0; ki < Nb[nlay]; ++ki)
				cout << "B[nlay][ki] " << B[nlay][ki] << endl;

			for (int mi = 0; mi < Nc[nlay]; ++mi)
				cout << "C[nlay][mi] " << C[nlay][mi] << endl;

		}


		if(0)
		if (nlay > 0) {
			//	for (int i = 0; i < k*n; ++i)
			//		*(B[nlay] + i) = 0;
			//else
			B[nlay] = C[nlay - 1];
			Nb[nlay] = Nc[nlay - 1];

			if (0) {
				cout << "And B again " << endl;

				for (int ki = 0; ki < Nb[nlay]; ++ki)
					cout << "B[nlay][ki] " << B[nlay][ki] << endl;
			}

		}
		// Nb muss auf die vorige gesetzt werden, man darf das aktuelle aber nicht verlieren.

	} // for (int nlay = 0; nlay < Nlay - 1; ++nlay) {

	cout << "Nnod = " << Nnod << endl;
	cout << "Nwij = " << Nwij << endl;

	trueVal = new double[top[Nlay - 1]];
	for (int ii = 0; ii < top[Nlay - 1]; ++ii) // lets initialize them just to avoid breakdowns
		trueVal[ii] = 0.0;

	err = new double*[Nlay];
	for (int nlay = 0; nlay < Nlay; ++nlay) {
//		nod[nlay] = new double[top[nlay] + 1]; // +1 is for D which is always 1.0
											   //	nod[nlay] = std::make_unique<double>(top[nlay] + 1);
		err[nlay] = new double[top[nlay]];
	}

	for (int nlay = 0; nlay < Nlay-1; ++nlay) {
		cout << "Na[nlay] "<< Na[nlay] << endl;
		cout << "Nb[nlay] "<< Nb[nlay] << endl;
		cout << "Nc[nlay] "<< Nc[nlay] << endl;
	}

	cout << "Neural Network is up and ready" << endl;

	return;

	//// some ml matrix trials
	//// multiplication
	//double * a = (double *)mkl_malloc(10 * sizeof(double), 64);
	//double * b = (double *)mkl_malloc(10 * sizeof(double), 64);
	//double * y = (double *)mkl_malloc(10 * sizeof(double), 64);

	//for (int i = 0; i < 10; ++i) {
	//	*(a + i) = (double)i;
	//	*(b + i) = (double)i;
	//}

	//vdMul(10, a, b, y);

	//for (int i = 0; i < 10; ++i)
	//	cout << y[i] << endl;

	//// dnn
	//dnnError_t error;
	//dnnLayout_t * pLayout = nullptr;
	//size_t dimension = 3;
	//const size_t size[] = { 2,3,1 };
	//const size_t strides[] = { 1,2,3 };
	//error = dnnLayoutCreate_F64(pLayout, dimension, size, strides);
	////cout << "elements search " << pLayout-> << endl;
	//
	//// dgemm taes max threads possible by default but can be restricted
	//cout << "mkl_get_max_threads() " << mkl_get_max_threads() << endl;
	//int th = 2;
	//mkl_set_num_threads(2);

}


void M::calc(bool doLearn) {

	for (int nlay = 0; nlay < Nlay - 1; ++nlay) { // Nlay - 1

		//m = 3, k = 2, n = 1;
		k = top[nlay] + D;
		m = top[nlay + 1];
		n = 1;

		if (nlay > 0) {
			for (int nb = 0; nb < Nb[nlay]; ++nb)
				B[nlay][nb] = C[nlay - 1][nb]; // in B fictive D is accounted for, in C it is not accounted for. 
											  // Thats because we stream from B's fictice D but not to C's fictive D
		}

		B[nlay][Nb[nlay] - 1] = 1.0; // Before we calc we need to do the D to 1.0

		for (int nc = 0; nc < Nc[nlay]; ++nc)
			C[nlay][nc] = 0.0; // Input values must all (!!!) be zero also the highest D,
							  // as a security since beta is 0.0 anyhow

							  // show ABC
		if (0) {

			cout << "calc nlay == " << nlay << endl;

			for (int mi = 0; mi < m; ++mi)
				for (int ki = 0; ki < k; ++ki)
					cout << "A[nlay][mi*k + ki] " << "A[" << nlay << "][" << mi << "*" << k << " + " << ki << "] " << A[0][mi*k + ki] << endl;

			for (int ki = 0; ki < Nb[nlay]; ++ki)
				cout << "B[nlay][ki] " << B[nlay][ki] << endl;

			for (int mi = 0; mi < Nc[nlay]; ++mi)
				cout << "C[nlay][mi] " << C[nlay][mi] << endl;
		}

		// manually calculating dgemm by Intel:
		//for (i = 0; i < m; i++) {
		//	for (j = 0; j < n; j++) {
		//		sum = 0.0;
		//		for (l = 0; l < k; l++)
		//			sum += A[k*i + l] * B[n*l + j];
		//		C[n*i + j] = sum;
		//	}
		//}

		//printf(" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			m, n, k, alpha, A[nlay], k, B[nlay], n, beta, C[nlay], n);
		//printf(" Computations completed.\n");

		// Squash C with e hoch minus 1
		// *p_val = 1.0 / (1 + pow(2.718, -1.0 * *p_val));
		if(1)
		for (int nc = 0; nc < Nc[nlay]; ++nc)
			C[nlay][nc] = 1.0 / (1.0 + pow(2.718, -1.0 * C[nlay][nc]));

		// Squash C with ReLU
		if(0)
		for (int nc = 0; nc < Nc[nlay] - 1; ++nc)
			C[nlay][nc] = C[nlay][nc] <= 0.0 ? 0.0 : C[nlay][nc];

		C[nlay][Nc[nlay] - 0] = 1.0; // D to One so it can serve as a fictive node. It sits on C but is not counted for in C[][Nc[nlay]]

		if (0)
			for (int nc = 0; nc < Nc[nlay]; ++nc)
				cout << "C["<<nlay<<"]["<<nc<<"] = " << C[nlay][nc] << endl;

	} // for

	if (!doLearn) {
		//getCalcRes(); // we denormalize only when we do not learn for efficiency
		//if (1) cout << "!doLearn" << endl;
		return;
	}

	//return;
	double test = 0.0;

	// Output layer errors, Errk = Ok * (1 - Ok) * (Tk - Ok) ....Error for the Output Nodes 
	for (int nt = 0; nt < top[Nlay - 1]; ++nt)
		err[Nlay - 1][nt] = //test--;
							C[Nlay - 2][nt] * // nod[Nlay - 1][n] *
							(1.0 - C[Nlay - 2][nt]) * // nod[Nlay - 1][n]) *
							(trueVal[nt] - C[Nlay - 2][nt]);  // nod[Nlay - 1][n]);

	if(0)
	for (int n = 0; n < top[Nlay - 1]; ++n)
		cout << "err[Nlay - 1][n] " << "err["<<Nlay - 1<<"]["<<n<<"]  "<< err[Nlay - 1][n] << endl;

	// Hidden layer errors, Erri = Oi * (1 - Oi) * SUM Errk*wik 
	for (int nlay = Nlay - 2; nlay > 0; --nlay) {

		//m = 3, k = 2, n = 1;
		k = top[nlay] + D;
		m = top[nlay + 1] + D;
		n = 1;

		for (int nt = 0; nt < top[nlay]; ++nt) {

			err[nlay][nt] = 0.0;

			//cout << "Hidden layer errors nlay " << nlay << " n " << n << endl;
			//cout << "err[" << nlay << "][" << n << "] " << err[nlay][n] << endl;

			for (int nnext = 0; nnext < top[nlay + 1]; ++nnext) {
				err[nlay][nt] +=
					err[nlay + 1][nnext] *
					A[nlay][nt+ nnext*k]; // [nlay][nnext*k]; // here we are. this is still wrong for more than one line
					// wij[nlay][n][nnext];
				//cout << "+= err[" << nlay << " + 1][" << nnext << "] * " << err[nlay + 1][nnext] << endl;
				//cout << "* A["<<nlay<<"]["<<nt<<"] " << A[nlay][nt] << endl;

			}

			//err[nlay][n] *= C[nlay-1][n] * (1.0 - C[nlay-1][n]); // or B[nlay][n]*(1.0 - B[nlay][n])
				// nod[nlay][n] * (1 - nod[nlay][n]);
			err[nlay][nt] *= B[nlay][nt] * (1.0 - B[nlay][nt]);
			//err[nlay][nt]=test--;

			if (0) cout << "err[nlay][n] " << " err[" << nlay << "][" << nt << "] " << err[nlay][nt] << endl;

		}
	}



	// wij's und d's ändern, 
	// wij = wij + learnRate * Errj * Oi
	// Dj  = Dj  + learnRate * Errj * 1.0, -> Dij = Dij + learnRate * Errj * 1.0 
	for (int nlay = 0; nlay < Nlay - 1; ++nlay) {

		k = top[nlay] + D;
		m = top[nlay + 1] + D;
		n = 1;

		//if (1) cout << "k " << k << " m  " << m << " n " << n << endl;

		for (int im = 0; im < m; ++im) {
			for (int ik = 0; ik < k - 1; ++ik) { // < ik -1 because the last one is d
				/* we do <= because we want to include the fictive d node which is always 1.0 */

				A[nlay][im*k + ik] // = test-- ;
					////wij[nlay - 1][nprev][n] 
					 += LearnRate *
					//// nod[nlay - 1][nprev] *
					B[nlay][ik] *
					err[nlay + 1][im];

				//cout << "A[" << nlay << "][" << im << "*" << k << " + " << ik << "] " << A[nlay][im*k + ik] << endl;
				//cout << "B[" << nlay << "][" << ik << "] * err[" << nlay << " + 1][" << ik << "] " << B[nlay][ik] << " * " << err[nlay + 1][ik] << endl;

			}
			// d's
			A[nlay][im*k + k - 1] += LearnRate * err[nlay + 1][im]; // = test--; //
			//cout << "D -> A[" << nlay << "][" << im << "*" << k << " + " << k << "-1] " << A[nlay][im*k + k - 1] << endl;
			//cout << "D -> err["<<nlay<<" + 1]["<<im<<"] " << err[nlay + 1][im] << endl;
		}
	}
	//auto cpu_engine = mkl_dnn::engine(mkl_dnn::engine::cpu, 0); // nope

	return ;

}
