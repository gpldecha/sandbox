#include "/home/guillaume/roscode/catkin_ws/src/machine_learning/include/machine_learning/ml/svminterface.h"
#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>
#include <iostream>


namespace kernel{

typedef float Qfloat;
typedef signed char schar;
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
static inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for(int t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix {
public:
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const = 0;
	virtual ~QMatrix() {}
};

class Kernel: public QMatrix {
public:
	Kernel(int l, svm_node * const * x, const svm_parameter& param);
	virtual ~Kernel();

	static double k_function(const svm_node *x, const svm_node *y,
				 const svm_parameter& param);
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const	// no so const...
	{
		swap(x[i],x[j]);
		if(x_square) swap(x_square[i],x_square[j]);
	}
protected:

	double (Kernel::*kernel_function)(int i, int j) const;

private:
	const svm_node **x;
	double *x_square;

	// svm_parameter
	const int kernel_type;
	const int degree;
	const double gamma;
	const double coef0;

	static double dot(const svm_node *px, const svm_node *py);
	double kernel_linear(int i, int j) const
	{
		return dot(x[i],x[j]);
	}
	double kernel_poly(int i, int j) const
	{
		return powi(gamma*dot(x[i],x[j])+coef0,degree);
	}
	double kernel_rbf(int i, int j) const
	{
		return exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
	}
	double kernel_sigmoid(int i, int j) const
	{
		return tanh(gamma*dot(x[i],x[j])+coef0);
	}
	double kernel_precomputed(int i, int j) const
	{
		return x[i][(int)(x[j][0].value)].value;
	}
};

Kernel::Kernel(int l, svm_node * const * x_, const svm_parameter& param)
:kernel_type(param.kernel_type), degree(param.degree),
 gamma(param.gamma), coef0(param.coef0)
{
	switch(kernel_type)
	{
		case LINEAR:
			kernel_function = &Kernel::kernel_linear;
			break;
		case POLY:
			kernel_function = &Kernel::kernel_poly;
			break;
		case RBF:
			kernel_function = &Kernel::kernel_rbf;
			break;
		case SIGMOID:
			kernel_function = &Kernel::kernel_sigmoid;
			break;
		case PRECOMPUTED:
			kernel_function = &Kernel::kernel_precomputed;
			break;
	}

	clone(x,x_,l);

	if(kernel_type == RBF)
	{
		x_square = new double[l];
		for(int i=0;i<l;i++)
			x_square[i] = dot(x[i],x[i]);
	}
	else
		x_square = 0;
}

Kernel::~Kernel()
{
	delete[] x;
	delete[] x_square;
}

double Kernel::dot(const svm_node *px, const svm_node *py)
{
	double sum = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}
	}
	return sum;
}

double Kernel::k_function(const svm_node *x, const svm_node *y,
			  const svm_parameter& param)
{
	switch(param.kernel_type)
	{
		case LINEAR:
			return dot(x,y);
		case POLY:
			return powi(param.gamma*dot(x,y)+param.coef0,param.degree);
		case RBF:
		{

			double sum = 0;
			while(x->index != -1 && y->index !=-1)
			{
				if(x->index == y->index)
				{
					double d = x->value - y->value;
					sum += d*d;
					++x;
					++y;
				}
				else
				{
					if(x->index > y->index)
					{
						sum += y->value * y->value;
						++y;
					}
					else
					{
						sum += x->value * x->value;
						++x;
					}
				}
			}

			while(x->index != -1)
			{
				sum += x->value * x->value;
				++x;
			}

			while(y->index != -1)
			{
				sum += y->value * y->value;
				++y;
			}
			return exp(-param.gamma*sum);
		}
		case SIGMOID:
			return tanh(param.gamma*dot(x,y)+param.coef0);
		case PRECOMPUTED:  //x: test (validation), y: SV
			return x[(int)(y->value)].value;
		default:
			return 0;  // Unreachable
	}
}


}

SvmInterface::SvmInterface():
v(22),
result(3)
{
	x 		   = NULL;
	dec_values = NULL;
	start	   = NULL;
	kvalue 	   = NULL;
	vote	   = NULL;
	//scaling_state = SCALING_ON;
	scalingfactor = 4000;
	scaling_state = SCALING_OFF;

	v[0]  = -0.380455;
	v[1]  = -0.322765;
	v[2]  = -0.306109;
	v[3]  = -0.394573;
	v[4]  = -0.185103;
	v[5]  = -0.304526;
	v[6]  =  0.0162907;
	v[7]  = -0.092825;
	v[8]  =  0.0416918;
	v[9]  = -0.0477526;
	v[10] =  0.576237;
	v[11] =  0.170755;
	v[12] =  0.217569;
	v[13] =  0.360726;
	v[14] =  0.190555;
	v[15] =  0.241528;
	v[16] = -0.0407609;
	v[17] =  0.0080335;
	v[18] = -0.00614907;
	v[19] =  0.914425;
	v[20] = -0.947381;
	v[21] = 0.101411;
}

void SvmInterface::test(){
	std::cout<< "in Test" << std::endl;
	std::cout<< "predict------------------> " << svmPredict(v) << std::endl;
}

SvmInterface::~SvmInterface(){
	if(x != NULL){
		free(x);
	}

	if(kvalue != NULL){
		free(kvalue);
	}

	if(start != NULL){
		free(start);
	}

	if(vote != NULL){
		free(vote);
	}
	if(sums != NULL){
		delete sums;
	}
}

void SvmInterface::loadParameters(const std::string& file){
	this->file = file;
	model = *svm_load_model(file.c_str());
	svm_allocate_k_memory(&model);
	printSVMinfo();
	sums = new double[model.nr_class];
}


double SvmInterface::svmPredict(const std::vector<double>& data){
	assert(x != NULL);
	//std::cout<< "data.size(): " << data.size() << std::endl;
	vec_double2svm_node(data,x);
	for(int i = 0 ; i < nbVar; i++){
		std::cout<< "x["<<i<<"]: " << x[i].value << " " << x[i].index << std::endl;
	}
	std::cout<<std::endl;


	double class_label = svm_predict_values_v2(&model,x,dec_values);

	return class_label;
}


void SvmInterface::setDimensionOfInput(int nbVar){
	x = (struct svm_node *) malloc((nbVar+1)*sizeof(struct svm_node));
	this->nbVar = nbVar+1;
}


void SvmInterface::vec_double2svm_node(const std::vector<double>& data,svm_node *x){

	assert(data.size() == (unsigned int)nbVar-1);
	int i = 0;
	for(i = 0; i < (int)data.size(); i++){
		x[i].index = i+1;
		if(scaling_state == SCALING_ON){
			x[i].value = data[i]/scalingfactor;
		}else{
			x[i].value = data[i];
		}
	}
	x[i].index = -1;
	x[i].value = -1;
}


void SvmInterface::print_x(){
	std::cout << " v = [";
	for(int i = 0; i < nbVar; i++){
		std::cout<< x[i].value << " ";
	}
	std::cout<< "]" << std::endl;
}


void SvmInterface::svm_allocate_k_memory(const svm_model* model){
	int l = model->l;
	int nr_class = model->nr_class;
	dec_values = Malloc(double, nr_class*(nr_class-1)/2);
	start	   = Malloc(int,nr_class);
	kvalue 	   = Malloc(double,l);
	vote 	   = Malloc(int,nr_class);
}


double SvmInterface::svm_predict_values_v2(const svm_model *model, const svm_node *x, double *prob_estimates){
	int i;
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
	{
		double *sv_coef = model->sv_coef[0];
		double sum = 0;
		for(i=0;i<model->l;i++)
			sum += sv_coef[i] * kernel::Kernel::k_function(x,model->SV[i],model->param);
		sum -= model->rho[0];
		*dec_values = sum;

		if(model->param.svm_type == ONE_CLASS)
			return (sum>0)?1:-1;
		else
			return sum;
	}
	else
	{
			int nr_class = model->nr_class;
			int l = model->l;

			for(i=0;i<l;i++){
				kvalue[i] = kernel::Kernel::k_function(x,model->SV[i],model->param);
			}

			start[0] = 0;
			for(i=1;i<nr_class;i++){
				start[i] = start[i-1]+model->nSV[i-1];
			}

			for(i=0;i<nr_class;i++){
				vote[i] = 0;
			}

			int p=0;
			for(i=0;i<nr_class;i++){
				for(int j=i+1;j<nr_class;j++)
				{
					double sum = 0;
					int si = start[i];
					int sj = start[j];
					int ci = model->nSV[i];
					int cj = model->nSV[j];

					int k;
					double *coef1 = model->sv_coef[j-1];
					double *coef2 = model->sv_coef[i];
					for(k=0;k<ci;k++){
						sum += coef1[si+k] * kvalue[si+k];
					}
					for(k=0;k<cj;k++){
						sum += coef2[sj+k] * kvalue[sj+k];
					}
					sum -= model->rho[p];
					dec_values[p] = sum;

					if(dec_values[p] > 0)
						++vote[i];
					else
						++vote[j];
					p++;
				}
			}

			int vote_max_idx = 0;

			for(i=1;i<nr_class;i++){
				if(vote[i] > vote[vote_max_idx]){
					vote_max_idx = i;
				}
			}

			return model->label[vote_max_idx];
	}

}

double SvmInterface::get_confidence(double *dec_values,int winner_id){

	double conf = 0;

	switch(winner_id)
	{
	case 0:

		conf = std::min(std::max(fabs(dec_values[0]),fabs(dec_values[1])),1.0);

		break;
	case 1:
		conf = std::min(std::max(fabs(dec_values[0]),fabs(dec_values[2])),1.0);

		break;
	case 2:
		conf = std::min(std::max(fabs(dec_values[1]),fabs(dec_values[2])),1.0);
		break;

	}

	return conf;
}

void SvmInterface::printSVMinfo(){

	std::cout<< " --- SVM info ---" << std::endl;


	std::cout<< "nr_class: " << model.nr_class << std::endl;
	std::cout<< "nSV: ";
	for(int i = 0; i < model.nr_class; i++){
		std::cout<< model.nSV[i] << " ";
	}
	std::cout<<std::endl;
	std::cout<< "type        -s: " << model.param.svm_type << std::endl;
	std::cout<< "kernel_type -t: " << model.param.kernel_type << std::endl;
	std::cout<< "gamma       -g: " << model.param.gamma << std::endl;
	std::cout<< "--------------" << std::endl;
}
