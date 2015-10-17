/*
 * matlabinterface.cpp
 *
 *  Created on: Sep 9, 2012
 *      Author: guillaume
 */




#include "/home/guillaume/roscode/catkin_ws/src/machine_learning/include/machine_learning/ml/matlabinterface.h"
#include <assert.h>


MatlabInterface::MatlabInterface():matlabWorkDirectory("'~/MatlabWorkSpace/'"),mxData_in(NULL),mxData_out(NULL),result(NULL){}


MatlabInterface::~MatlabInterface(){

	mxDestroyArray(mxData_out);
	mxDestroyArray(mxData_in);

	if(result != NULL){
		delete [] *result;
		delete [] result;
	}
	if(Data_in != NULL){

	     for ( int r = 0; r < nbVar_in; ++r ){
	           delete [] Data_in[r];
	     }
	           delete [] Data_in;
	}

	if(Data_out != NULL){
		delete [] Data_in;
	}
}


void MatlabInterface::setPathToWorkingDir(const std::string& path){
	matlabWorkDirectory = path;
}

void MatlabInterface::startMatlab(){
	engine = engOpen("\0");
	assert(engine != 0);
	engEvalString(engine, "clear all;");
	std::string matlab_path = "addpath(genpath(";
	matlab_path.append(matlabWorkDirectory).append("));");
	engEvalString(engine, matlab_path.c_str());
	engEvalString(engine, "cd('~/MatlabWorkSpace/biotouch2');");

}

void MatlabInterface::setDataSizeIn(int nbSamples,int nbVar){
	this->nbVar_out = nbVar;
	this->nbSamples_out = nbSamples;

	result = new double*[nbVar];
	for(int i = 0; i < nbVar; i++){
		result[i] = new double[nbSamples];
	}
}

void MatlabInterface::setDataSizeOut(int nbSamples,int nbVar){
	this->nbVar_in = nbVar;
	this->nbSamples_in = nbSamples;
	mxData_in = mxCreateDoubleMatrix(nbVar,nbSamples,mxREAL);

	Data_in = new double*[nbVar];
	for(int i = 0; i < nbVar; i++){
		Data_in[i] = new double[nbSamples];
	}
}

void MatlabInterface::inputVar(const std::string& varName,double **data){
	 if(mxData_in != NULL){
		 memcpy((void *) mxGetPr(mxData_in), (void *)data, sizeof(double)*nbVar_in*nbSamples_in);
		 engPutVariable(engine, varName.c_str(), mxData_in);
	 }else{
		std::cout<< "mxData_in is NULL: you call method: setDataSizeIn(int nbSamples,int nbVar)" << std::endl;
	 }
}

void MatlabInterface::outPutVar(const std::string& varName,double **data){
	if(mxData_out != NULL){
	 Data_out = mxGetPr(mxData_out);
	}else{
		std::cout<< "mxData_out is NULL: you call method: setDataSizeOut(int nbSamples,int nbVar)" << std::endl;
	}
	 // get data and D_out and put it in result;

}

void MatlabInterface::rowMajorToColMajor( double *array1,double **array2,int nbRow, int nbCol){
	for(int r = 0;r < nbRow;r++){
		for(int c = 0; c < nbCol;c++){
			array2[r][c] = array1[r + c*nbRow];
		}
	}
}

void MatlabInterface::colMajorToRowMajor( double *array1,double **array2,int nbRow, int nbCol){
	for(int r = 0;r < nbRow;r++){
		for(int c = 0; c < nbCol;c++){
			array2[r][c] = array1[r*nbCol + c];
		}
	}
}

void MatlabInterface::vector2matArr(const std::vector<double>& data, double **mat_arr){
	for(unsigned int i = 0; i < data.size(); i++){
		mat_arr[i][0] = data[i];
	}
}


