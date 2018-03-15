
#include "N.h" 
#include "MN.h"

//#include <stdio.h>
//#include <stdlib.h>
//#include <boost/numeric/ublas/matrix.hpp>

//template<typename T> ft1(T x) {
//	if (typename T == int)
//		return 2 * x;
//	else return x;
//}



int f1(std::unique_ptr<N> rp) {

	return 0;
}


std::unique_ptr<U> f2(std::unique_ptr<U> rp) {

	std::unique_ptr<U> rx = std::move(rp);

	return std::move(rx);
	//return rx; // works as well, maybe because of RVO
}


int main0() {

	double * sd = new double();
	*sd = 5;
	std::unique_ptr<double> vd = nullptr; // pointer to a single double
	vd = std::make_unique<double>(10);   // one double is initialized to 10.0;
	cout << "vd = " << *vd << endl;

	double * pd = new double[10];
	for (int i = 0; i < 10; ++i)
		pd[i] = 5;
	std::unique_ptr<double[]> ud = nullptr;
	ud = std::make_unique<double[]>(10); // 10 doubles are initialized to 0.0;
	if (0)
		for (int i = 0; i < 10; ++i)
			cout << ud[i] << endl;

	double ** ppd = new double *[10];
	for (int i = 0; i < 10; ++i)
		ppd[i] = new double[5];
	for (int i = 0; i < 10; ++i)
		for (int j = 0; j < 5; ++j)
			ppd[i][j] = 27;
	std::unique_ptr<std::unique_ptr<double[]>[]> uud = nullptr;
	uud = std::make_unique<std::unique_ptr<double[]>[]>(10);
	for (int i = 0; i < 10; ++i)
		uud[i] = std::make_unique<double[]>(10); //  std::make_unique<double[]>(5);

	if (0)
		for (int i = 0; i < 10; ++i)
			for (int j = 0; j < 10; ++j)
				cout << "uud[" << i << "][" << j << "] = " << uud[i][j] << endl;

	double *** pppd = new double **[10]; //.......
	std::unique_ptr<std::unique_ptr<std::unique_ptr<double[]>[]>[]> uuud = nullptr;
	uuud = std::make_unique<std::unique_ptr<std::unique_ptr<double[]>[]>[]>(10);
	for (int i = 0; i < 10; ++i) {
		uuud[i] = std::make_unique<std::unique_ptr<double[]>[]>(10);
		for (int j = 0; j < 10; ++j)
			uuud[i][j] = std::make_unique<double[]>(10);
	}

	for (int i = 0; i < 10; ++i)
		for (int j = 0; j < 10; ++j)
			for (int k = 0; k < 10; ++k)
				cout << "uuud[" << i << "][" << j << "][" << k << "] = " << uuud[i][j][k] << endl;
	return 0;
}


int main1(){
	//create the Network on stack
	//N n{ topologie{ 2,3,1 }, learnRate{ 0.9 }, activationMethodchoosen::eins_durch_ehoch, 
	// normalization{ 1.0, 0.0, 1.0, 0.0 } };

	// C++17, on heap with smart pointer
	std::unique_ptr<N> p(new N{ topologie{ 2,3,1 }, learnRate{ 0.9 }, activationMethodchoosen::eins_durch_ehoch,
		normalization{ 1.0, 0.0, 1.0, 0.0 } });

	// C++17
	if(0)
	std::unique_ptr<N> p1 = std::make_unique<N> ( topologie{ 2,3,1 }, learnRate{ 0.09 }, 
		activationMethodchoosen::ReLU, // learn 0.09, random 0, 1, iterations 500, 1000
		normalization{ 1.0, 0.0, 1.0, 0.0 }, randomInit { 0.0, 1.0 }); // Attention not {} but () for parameter list

	//// C++14 // works 
	//std::unique_ptr<U> p  = std::make_unique<U>(topologie{ 2, 3, 1 }, learnRate{ 0.9 }, 
	//	activationMethodchoosen::eins_durch_ehoch,
	//	normalization{ 1.0, 0.0, 1.0, 0.0 }); // Attention not {} but () for parameter list

	//// C++14 // works 
	//std::unique_ptr<U> pu = std::make_unique<U>(topologie{ 2, 3, 1 }, learnRate{ 0.9 },
	//	activationMethodchoosen::eins_durch_ehoch,
	//	normalization{ 1.0, 0.0, 1.0, 0.0 }); // Attention not {} but () for parameter list

	//std::unique_ptr<N[]> vu1(new N[10]( topologie{ 2,3,1 }, learnRate{ 0.9 }, activationMethodchoosen::eins_durch_ehoch,
	//	normalization{ 1.0, 0.0, 1.0, 0.0 } )); // array of unique pointer does not allow parameters in the constructor

	std::chrono::high_resolution_clock::time_point timex;
	std::chrono::nanoseconds elapsed;
	// 1 sec == 1 000 000 000 Nanosekunden

	//inpPtr inp = p->input;

	timex = std::chrono::high_resolution_clock::now();
	if (1) // proper sequence for teaching
		for (int it = 0; it < 1000; ++it) {
			p->input[0] = 0.0;
			p->input[1] = 0.0; // p->input[0] = 0.0;
			p->trueVal[0] = 0.0;
			p->calc(learn{ true });
			p->input[0] = 0.0;
			p->input[1] = 1.0;
			p->trueVal[0] = 1.0;
			p->calc(learn{ true });
			p->input[0] = 1.0;
			p->input[1] = 0.0;
			p->trueVal[0] = 1.0;
			p->calc(learn{ true });
			p->input[0] = 1.0;
			p->input[1] = 1.0;
			p->trueVal[0] = 0.0;
			p->calc(learn{ true });
		} // for (int it = 0; it < 1000; ++it) {
	elapsed = std::chrono::high_resolution_clock::now() - timex;


	if (1) { // check if the network really learned to solve its task
		cout << "Ergebnis:" << endl;
		p->input[0] = 0.0;
		p->input[1] = 0.0;
		p->calc(learn{ false });
		cout << p->input[0] << p->input[1] << " -> " << p->output[0] << endl; 
		p->input[0] = 0.0;
		p->input[1] = 1.0;
		p->calc(learn{ false });
		cout << p->input[0] << p->input[1] << " -> " << p->output[0] << endl; 
		p->input[0] = 1.0;
		p->input[1] = 0.0;
		p->calc(learn{ false });
		cout << p->input[0] << p->input[1] << " -> "  << p->output[0] << endl; 
		p->input[0] = 1.0;
		p->input[1] = 1.0;
		p->calc(learn{ false });
		cout << p->input[0] << p->input[1] << " -> " << p->output[0] << endl; // " " << (p->getCalcRes())[0] << endl;
	} // if (1) { // check if the network really learned to solve its task

	std::cout << "Elapsed time in seconds = " << elapsed.count() / 1000000000.0 << std::endl;
	
	//f1(std::move(p));

	//std::unique_ptr<U> r = f2(std::move(p));

	//r->input[0] = 1.0;
	//r->input[1] = 0.0;
	//r->calc(learn{ false });
	//cout << r->input[0] << r->input[1] << " -> " << r->output[0] << endl; // " " << (p->getCalcRes())[0] << endl;

	//getchar();
	return 0;
}


#define min(x,y) (((x) < (y)) ? (x) : (y))


//int matrix0() {
//
//	// Multiplying Matrices Using dgemm
//	//
//	// Attention:
//	// libiomp5md.dll has to be where the exe is
//	// has 1.55kb and is located in
//	// D : \IntelSWTools\compilers_and_libraries_2018.1.156\windows\redist\intel64_win\compiler
//
//
//	double *A, *B, *C;
//	int m, n, k, i, j;
//	double alpha, beta;
//
//	printf("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
//		" Intel(R) MKL function dgemm, where A, B, and  C are matrices and \n"
//		" alpha and beta are double precision scalars\n\n");
//
//	m = 2000, k = 200, n = 1000;
//
//
//	printf(" Initializing data for matrix multiplication C=A*B for matrix \n"
//		" A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);
//
//	alpha = 1.0; beta = 0.0;
//
//	printf(" Allocating memory for matrices aligned on 64-byte boundary for better \n"
//		" performance \n\n");
//	A = (double *)mkl_malloc(m*k * sizeof(double), 64);
//	B = (double *)mkl_malloc(k*n * sizeof(double), 64);
//	C = (double *)mkl_malloc(m*n * sizeof(double), 64);
//
//	if (A == NULL || B == NULL || C == NULL) {
//		printf("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
//		mkl_free(A);
//		mkl_free(B);
//		mkl_free(C);
//		return 1;
//	}
//
//		printf(" Intializing matrix data \n\n");
//		for (i = 0; i < (m*k); i++) {
//			A[i] = (double)(i + 1);
//		}
//
//		for (i = 0; i < (k*n); i++) {
//			B[i] = (double)(-i - 1);
//		}
//
//		for (i = 0; i < (m*n); i++) {
//			C[i] = 0.0;
//		}
//
//
//		//void cblas_dgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, 
//		//	const MKL_INT m, const MKL_INT n, const MKL_INT k, 
//		//	const double alpha, const double *a, const MKL_INT lda, const double *b, const MKL_INT ldb, 
//		//	const double beta, double *c, const MKL_INT ldc);
//
//
//		printf(" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
//		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//			m, n, k, alpha, A, k, B, n, beta, C, n);
//		printf("\n Computations completed.\n\n");
//
//		printf(" Top left corner of matrix A: \n");
//		for (i = 0; i<min(m, 6); i++) {
//			for (j = 0; j<min(k, 6); j++) {
//				printf("%12.0f", A[j + i * k]);
//			}
//			printf("\n");
//		}
//
//		printf("\n Top left corner of matrix B: \n");
//		for (i = 0; i<min(k, 6); i++) {
//			for (j = 0; j<min(n, 6); j++) {
//				printf("%12.0f", B[j + i * n]);
//			}
//			printf("\n");
//		}
//
//		printf("\n Top left corner of matrix C: \n");
//		for (i = 0; i<min(m, 6); i++) {
//			for (j = 0; j<min(n, 6); j++) {
//				printf("%12.5G", C[j + i * n]);
//			}
//			printf("\n");
//		}
//
//		printf("\n Deallocating memory \n\n");
//		mkl_free(A);
//		mkl_free(B);
//		mkl_free(C);
//
//		printf(" Example completed. \n\n");
//
//	return 0;
//}
//
//
//int matrix1() {
//
//	// Multiplying Matrices Using dgemm
//	//
//	// Attention:
//	// libiomp5md.dll has to be where the exe is
//	// has 1.55kb and is located in
//	// D : \IntelSWTools\compilers_and_libraries_2018.1.156\windows\redist\intel64_win\compiler
//
//
//	double *A, *B, *C;
//	int m, n, k, i, j;
//	double alpha, beta;
//
//	printf("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
//		" Intel(R) MKL function dgemm, where A, B, and  C are matrices and \n"
//		" alpha and beta are double precision scalars\n\n");
//
//	m = 3, k = 2, n = 1;
//
//
//	printf(" Initializing data for matrix multiplication C=A*B for matrix \n"
//		" A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);
//
//	alpha = 1.0; beta = 0.0;
//
//	printf(" Allocating memory for matrices aligned on 64-byte boundary for better \n"
//		" performance \n\n");
//	A = (double *)mkl_malloc(m*k * sizeof(double), 64);
//	B = (double *)mkl_malloc(k*n * sizeof(double), 64);
//	C = (double *)mkl_malloc(m*n * sizeof(double), 64);
//
//	if (A == NULL || B == NULL || C == NULL) {
//		printf("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
//		mkl_free(A);
//		mkl_free(B);
//		mkl_free(C);
//		return 1;
//	}
//
//	printf(" Intializing matrix data \n\n");
//	for (i = 0; i < (m*k); i++) {
//		A[i] = (double)(i + 1);
//	}
//
//	for (i = 0; i < (k*n); i++) {
//		B[i] = (double)(-i - 1);
//	}
//
//	for (i = 0; i < (m*n); i++) {
//		C[i] = 0.0;
//	}
//
//
//	//  void cblas_dgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
//	//	const MKL_INT m, const MKL_INT n, const MKL_INT k,
//	//	const double alpha, const double *a, const MKL_INT lda, const double *b, const MKL_INT ldb,
//	//	const double beta, double *c, const MKL_INT ldc);
//
//
//	printf(" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
//	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//		m, n, k, alpha, A, k, B, n, beta, C, n);
//	printf("\n Computations completed.\n\n");
//
//	printf(" Top left corner of matrix A: \n");
//	for (i = 0; i<min(m, 6); i++) {
//		for (j = 0; j<min(k, 6); j++) {
//			printf("%12.0f", A[j + i * k]);
//		}
//		printf("\n");
//	}
//
//	printf("\n Top left corner of matrix B: \n");
//	for (i = 0; i<min(k, 6); i++) {
//		for (j = 0; j<min(n, 6); j++) {
//			printf("%12.0f", B[j + i * n]);
//		}
//		printf("\n");
//	}
//
//	printf("\n Top left corner of matrix C: \n");
//	for (i = 0; i<min(m, 6); i++) {
//		for (j = 0; j<min(n, 6); j++) {
//			printf("%12.5G", C[j + i * n]);
//		}
//		printf("\n");
//	}
//
//	printf("\n Deallocating memory \n\n");
//	mkl_free(A);
//	mkl_free(B);
//	mkl_free(C);
//
//	printf(" Example completed. \n\n");
//
//	return 0;
//}


int mainMKL() {

	
	std::unique_ptr<M> p = std::make_unique<M>(topologie{ 2, 3, 1 }, learnRate{ 0.9 }, activationMethodchoosen::eins_durch_ehoch,
		normalization{ 1.0, 0.0, 1.0, 0.0 });

	//return 0;

	std::chrono::high_resolution_clock::time_point timex;
	std::chrono::nanoseconds elapsed;
	// 1 sec == 1 000 000 000 Nanosekunden

	timex = std::chrono::high_resolution_clock::now();

	if (1) // proper sequence for teaching
		for (int it = 0; it < 1000; ++it) {

			p->B[0][0] = 0.0;
			p->B[0][1] = 0.0;
			p->trueVal[0] = 0.0;
			p->calc(true);
			
			//cout << "break " << endl;
			//break;

			p->B[0][0] = 0.0;
			p->B[0][1] = 1.0;
			p->trueVal[0] = 1.0;
			p->calc(true);

			p->B[0][0] = 1.0;
			p->B[0][1] = 0.0;
			p->trueVal[0] = 1.0;
			p->calc(true);

			p->B[0][0] = 1.0;
			p->B[0][1] = 1.0;
			p->trueVal[0] = 0.0;
			p->calc(true);

		}

	////return 0;

	elapsed = std::chrono::high_resolution_clock::now() - timex;

	p->B[0][0] = 0.0;
	p->B[0][1] = 0.0;
	p->calc(false);
	cout << "0 0 calc result p->C[1][0] " << (p->C[1][0]) << endl;

	//return 0;

	p->B[0][0] = 0.0;
	p->B[0][1] = 1.0;
	p->calc(false);
	cout << "0 1 calc result p->C[1][0] " << (p->C[1][0]) << endl;

	p->B[0][0] = 1.0;
	p->B[0][1] = 0.0;
	p->calc(false);
	cout << "1 0 calc result p->C[1][0] " << (p->C[1][0]) << endl;

	p->B[0][0] = 1.0;
	p->B[0][1] = 1.0;
	p->calc(false);
	cout << "1 1 calc result p->C[1][0] " << (p->C[1][0]) << endl;

	//cout << "calc result p->A[0][0] " << (p->A[0][0]) << endl;
	//cout << "calc result p->A[0][1] " << (p->A[0][1]) << endl;
	//cout << "calc result p->C[1][0] " << (p->C[1][0]) << endl;
	//cout << "calc result p->C[1][1] " << (p->C[1][1]) << endl;

	std::cout << "Elapsed time in seconds = " << elapsed.count() / 1000000000.0 << std::endl;

	return 0;
}


int aLittleIteratorTry() {

	std::vector<double> v{ 1,2,3,4,5,6,7 };

	for (std::vector<double>::iterator p = v.begin(); p != v.end(); ++p)
		cout << *p << endl;
	
	for (auto e : v)
		cout << e << endl;






	return 0;
}


int f() {

	return 5;
}


int mainMN() {

	std::unique_ptr<MN> p = std::make_unique<MN>(topologie{ 2, 3, 1 }, learnRate{ 0.9 }, activationMethodchoosen::eins_durch_ehoch,
		normalization{ 1.0, 0.0, 1.0, 0.0 }, randomInit { -1.0, 1.0 });

	//std::unique_ptr<N> p = std::make_unique<N>(topologie{ 2,3,1 }, learnRate{ 0.09 },
	//	activationMethodchoosen::ReLU, // learn 0.09, random 0, 1, iterations 500, 1000
	//	normalization{ 1.0, 0.0, 1.0, 0.0 }, randomInit{ 0.0, 1.0 }); // Attention not {} but () for parameter list

	std::chrono::high_resolution_clock::time_point timex;
	std::chrono::nanoseconds elapsed;
	// 1 sec == 1 000 000 000 Nanosekunden

	//inpPtr inp = p->input;

	timex = std::chrono::high_resolution_clock::now();
	if (1) // proper sequence for teaching
		for (int it = 0; it < 1000; ++it) {
			p->input[0] = 0.0;
			p->input[1] = 0.0; // p->input[0] = 0.0;
			p->trueVal[0] = 0.0;
			p->calc(learn{ true });
			p->input[0] = 0.0;
			p->input[1] = 1.0;
			p->trueVal[0] = 1.0;
			p->calc(learn{ true });
			p->input[0] = 1.0;
			p->input[1] = 0.0;
			p->trueVal[0] = 1.0;
			p->calc(learn{ true });
			p->input[0] = 1.0;
			p->input[1] = 1.0;
			p->trueVal[0] = 0.0;
			p->calc(learn{ true });
		} // for (int it = 0; it < 1000; ++it) {
	elapsed = std::chrono::high_resolution_clock::now() - timex;


	if (1) { // check if the network really learned to solve its task
		cout << "Ergebnis:" << endl;
		p->input[0] = 0.0;
		p->input[1] = 0.0;
		p->calc(learn{ false });
		cout << p->input[0] << p->input[1] << " -> " << p->output[0] << endl;
		p->input[0] = 0.0;
		p->input[1] = 1.0;
		p->calc(learn{ false });
		cout << p->input[0] << p->input[1] << " -> " << p->output[0] << endl;
		p->input[0] = 1.0;
		p->input[1] = 0.0;
		p->calc(learn{ false });
		cout << p->input[0] << p->input[1] << " -> " << p->output[0] << endl;
		p->input[0] = 1.0;
		p->input[1] = 1.0;
		p->calc(learn{ false });
		cout << p->input[0] << p->input[1] << " -> " << p->output[0] << endl; // " " << (p->getCalcRes())[0] << endl;
	} // if (1) { // check if the network really learned to solve its task

	std::cout << "Elapsed time in seconds = " << elapsed.count() / 1000000000.0 << std::endl;

	return 0;
}

int mainMKL1() {

	std::unique_ptr<MKL> p = std::make_unique<MKL>(topologie{ 2, 3, 1 }, learnRate{ 0.9 }, activationMethodchoosen::eins_durch_ehoch,
		normalization{ 1.0, 0.0, 1.0, 0.0 }, randomInit{ -1.0, 1.0 });

	//std::unique_ptr<N> p = std::make_unique<N>(topologie{ 2,3,1 }, learnRate{ 0.09 },
	//	activationMethodchoosen::ReLU, // learn 0.09, random 0, 1, iterations 500, 1000
	//	normalization{ 1.0, 0.0, 1.0, 0.0 }, randomInit{ 0.0, 1.0 }); // Attention not {} but () for parameter list

	std::chrono::high_resolution_clock::time_point timex;
	std::chrono::nanoseconds elapsed;
	// 1 sec == 1 000 000 000 Nanosekunden

	//inpPtr inp = p->input;

	timex = std::chrono::high_resolution_clock::now();
	if (1) // proper sequence for teaching
		for (int it = 0; it < 1000; ++it) {
			p->input[0] = 0.0;
			p->input[1] = 0.0; // p->input[0] = 0.0;
			p->trueVal[0] = 0.0;
			p->calc(learn{ true });
			p->input[0] = 0.0;
			p->input[1] = 1.0;
			p->trueVal[0] = 1.0;
			p->calc(learn{ true });
			p->input[0] = 1.0;
			p->input[1] = 0.0;
			p->trueVal[0] = 1.0;
			p->calc(learn{ true });
			p->input[0] = 1.0;
			p->input[1] = 1.0;
			p->trueVal[0] = 0.0;
			p->calc(learn{ true });
		} // for (int it = 0; it < 1000; ++it) {
	elapsed = std::chrono::high_resolution_clock::now() - timex;


	if (1) { // check if the network really learned to solve its task
		cout << "Ergebnis:" << endl;
		p->input[0] = 0.0;
		p->input[1] = 0.0;
		p->calc(learn{ false });
		cout << p->input[0] << p->input[1] << " -> " << p->output[0] << endl;
		p->input[0] = 0.0;
		p->input[1] = 1.0;
		p->calc(learn{ false });
		cout << p->input[0] << p->input[1] << " -> " << p->output[0] << endl;
		p->input[0] = 1.0;
		p->input[1] = 0.0;
		p->calc(learn{ false });
		cout << p->input[0] << p->input[1] << " -> " << p->output[0] << endl;
		p->input[0] = 1.0;
		p->input[1] = 1.0;
		p->calc(learn{ false });
		cout << p->input[0] << p->input[1] << " -> " << p->output[0] << endl; // " " << (p->getCalcRes())[0] << endl;
	} // if (1) { // check if the network really learned to solve its task

	std::cout << "Elapsed time in seconds = " << elapsed.count() / 1000000000.0 << std::endl;

	return 0;
}


int main() {

	//cout << ft1(5) << endl; // does not work

	//main0();
	main1(); // works, // great original Version, pure pointer
	//matrix0();
	//matrix1(); 
	mainMKL(); // works now but is a complicated naming convention and entirely for MKL
	//mainMN(); // works, own matrix
	mainMKL1(); // works // Good version, matrix, own matrix and mkl matrix,, own malloc and intel malloc
	//aLittleIteratorTry();

	getchar();
	return 0;
}