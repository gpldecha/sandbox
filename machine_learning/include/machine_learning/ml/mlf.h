/*
 * wcov.h
 *
 *  Created on: Aug 30, 2012
 *      Author: guillaume
 */

#ifndef MLF_H_
#define MLF_H_

#include <armadillo>
#include <utility>
#include <cmath>
#include "global.h"

namespace mlf{

static arma::mat wcov(const arma::mat& X,const arma::vec w){

	int T = X.n_rows;
	int N = X.n_cols;
	arma::mat I(N,N);
	I.eye();
	arma::mat C = X - repmat(w.st()*X,T,1);
	C = C.st() * (C % repmat(w,1,N));
	C = 0.5 * (C + C.st()) + I*0.0001;
	return C;
}


static std::pair<double,double> N(const arma::colvec2& x,const arma::colvec2& mean, const arma::mat22& covariance){
	std::pair<double,double> out;
    double denom = 1/(2*M_PI*sqrt(det(covariance)));
    arma::colvec2 X = x - mean;
    arma::mat quadratic = X.st()*inv(covariance)*X;
    double nom = exp( -0.5*quadratic(0,0) );
    out.first = denom*nom;
    out.second = denom;
    return out;
}


}

#endif /* MLF_H_ */
