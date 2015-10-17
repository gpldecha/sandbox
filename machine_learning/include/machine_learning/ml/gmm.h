/*
 * gmm.h
 *
 *  Created on: Nov 9, 2012
 *      Author: guillaume
 */

#ifndef GMM_H_
#define GMM_H_

#include "matlabinterface.h"
#include <vector>
#include <string>

class Gmm : public MatlabInterface{

public:

	Gmm();

	void loadGmm();

	void computeMarginal(const std::vector<double>& data);


private:

	std::string varInName;

};


#endif /* GMM_H_ */
