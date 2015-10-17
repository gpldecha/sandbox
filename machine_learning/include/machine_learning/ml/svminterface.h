#ifndef SVMINTERFACE_H_
#define SVMINTERFACE_H_

#include "svm.h"
#include <string>
#include <vector>



class SvmInterface {

	enum scaling {SCALING_ON,SCALING_OFF};

public:

	SvmInterface();

	~SvmInterface();

	void setDimensionOfInput(int nbVar);

	void loadParameters(const std::string& file);

	double svmPredict(const std::vector<double>& data);

	void test();

private:

	void vec_double2svm_node(const std::vector<double>& data,svm_node *x);

	double get_confidence(double *dec_values, int winner_id);

	void printSVMinfo();

	void print_x();

	void svm_allocate_k_memory(const svm_model* model);

	double svm_predict_values_v2(const svm_model *model, const svm_node *x, double *prob_estimates);

private:

	std::string file;
	int nbClasses;
	int nbVar;
	std::vector<double> v;
	svm_model model;
	struct svm_node *x;

	std::vector<std::pair<int,double> > result;

	scaling scaling_state;
	double scalingfactor;


	double  *sums;
	double 	*kvalue;
	double 	*dec_values;
	int 	*start;
	int 	*vote;

};

#endif /* SVMINTERFACE_H_ */

