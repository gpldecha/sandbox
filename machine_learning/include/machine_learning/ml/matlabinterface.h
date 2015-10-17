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

public:

	MatlabInterface();

	~MatlabInterface();

	void setPathToWorkingDir(const std::string& path);

	void startMatlab();

	void setDataSizeIn(int nbSamples,int nbVar);

	void setDataSizeOut(int nbSamples, int nbVar);

	void inputVar(const std::string& varName,double **data);

	void outPutVar(const std::string& varName,double **data);


protected:

	 void rowMajorToColMajor(double *array1,double **array2,int nbRows,int nbCols);

	 void colMajorToRowMajor(double *array1,double **array2,int nbRows,int nbCols);

	 void vector2matArr(const std::vector<double>& data, double **mat_arr);

protected:

	Engine *engine;
	mxArray *mxData_in;
	mxArray *mxData_out;

	double **Data_in;
	double *Data_out;

	std::string matlabWorkDirectory;

	int nbVar_in, nbVar_out;
	int nbSamples_in, nbSamples_out;
	std::string purpose;

public:

	double **result;

};



#endif /* MATLABINTERFACE_H_ */
