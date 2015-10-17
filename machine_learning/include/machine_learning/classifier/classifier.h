#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#include <armadillo>
#include <vector>
#include <string>

namespace ml {

class Classifier{

public:

    Classifier();

    virtual ~Classifier();

    virtual int predict(const arma::colvec& X) = 0;

    arma::colvec& get_probabilities();

    std::size_t get_num_classes();

    std::size_t get_data_size();

    std::vector<std::string>& get_class_labels();

protected:

    arma::colvec                prob_classes;
    std::size_t                 num_classes;
    std::size_t                 data_size;
    std::vector<std::string>    class_labels;


};

}

#endif
