#ifndef CLASSIFIER_INTERFACE_H_
#define CLASSIFIER_INTERFACE_H_

#include "classifier/classifier.h"

namespace ml {

class Classifier_interface{

public:

    typedef enum {GMM,SVM} classifer_type;

public:

    Classifier_interface(const std::string& type,std::string path_to_parameters);

    ~Classifier_interface();

    int predict(const arma::colvec& x);

    void get_probability(arma::colvec& prob);

public:

    ml::Classifier* ptr_classifier;

};

}

#endif
