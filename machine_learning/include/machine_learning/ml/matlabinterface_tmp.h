/*
 * matlabinterface.h
 *
 *  Created on: Sep 9, 2012
 *      Author: guillaume
 */

#ifndef MATLABINTERFACE_H_
#define MATLABINTERFACE_H_

#include "engine.h"
#include "global.h"
#include <assert.h>
#include <string>
#include <armadillo>


class MatlabInterface{

	enum toolbox {murphy,pmtk3,bnt};

public:

	MatlabInterface(std::string purpose);

	~MatlabInterface();

	void setPathToWorkingDir(const std::string& path);

	void startMatlab();

	void set_data_size_in(int nbSamples,int nbVar);

	void set_data_size_out(int nbSamples, int nbVar);

	void matlab_compare_hmm_Models(const vector_double_buffers& buffer);

	void matlab_compute_marginal(const std::vector<double>& data);




private:

	void buffer2array(const vector_double_buffers& buffer,double **array,int rowSize, int colSize);

	void loadHmm();

	void loadGmm();

	void cArr2matArr(double *mat_arr, double **c_arr,const int nbSamples, const int nbVar);

	void vector2matArr(const std::vector<double>& data, double *mat_arr);

	void matArr2cArr(const double *mat_arr, double **c_arr,const int nbSamples, const int nbVar);
	void matArr2cArr(const double *mat_arr, double *c_arr, const int nbVar);

	void print(double **array,int nbRows, int nbCols);

	void printClass(double **array, int nbRows, int nbCols);

private:

	Engine *engine;
	mxArray *mxData_in;
	mxArray *mxData_out;

	double **Data_in;
	double *Data_in2;
	double *Data_out;

	arma::colvec3 test;

	std::string matlabWorkDirectory;

	int nbVar_in, nbVar_out;
	int nbSamples_in, nbSamples_out;
	std::string purpose;

public:

	double **result;
	double  *result2;

};



#endif /* MATLABINTERFACE_H_ */
