#include "machine_learning/classifier/classifier.h"

namespace ml {

Classifier::Classifier()
{
    num_classes = 0;
    prob_classes.resize(1);
}

Classifier::~Classifier(){

}

arma::colvec& Classifier::get_probabilities(){
    return prob_classes;
}

std::size_t Classifier::get_num_classes(){
    return  num_classes;
}

std::size_t Classifier::get_data_size(){
    return data_size;
}

std::vector<std::string>& Classifier::get_class_labels(){
    return class_labels;
}



}
