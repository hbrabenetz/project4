#pragma once

#include <iostream>
#include <cstdio>
#include <vector>
#include <initializer_list>
#include <cmath>
#include <random>
#include <functional>
#include <chrono>
#include <tuple>
#include <memory> // for unique_pointer
#include <utility> // for std::move of unique pointer
#include <mkl.h>

using namespace std;

using topologie = std::initializer_list<int>;

enum class activationMethodchoosen { tanh_sigmoid, eins_durch_ehoch, ReLU, no_formula }; // Attention, it always needs an activation method, 

inline void eins_durch_ehoch(double * p_val);
inline double derivative_eins_durch_ehoch(double * p_val);

inline void ReLU(double * p_val);
inline double derivative_ReLU(double * p_val);


using normalization = std::tuple<double, double, double, double>;
using randomInit = std::tuple<double, double>;
using learnRate = double;
using learn = bool;

//using input = std::unique_ptr<std::unique_ptr<double[]>[]> ;
//using inpPtr = std::unique_ptr<double[]>&;
using inpPtr = std::unique_ptr<double[]>;


