#include "machine_learning/node/classifier_interface.h"

#include "machine_learning/classifier/gmm_classifier.h"

namespace ml{

Classifier_interface::Classifier_interface(const std::string& type,std::string path_to_parameters){

    ptr_classifier = NULL;

    if(type == "gmm"){
        ptr_classifier = new ml::Gmm_classifier(path_to_parameters);
    }else{

    }

}

Classifier_interface::~Classifier_interface(){
    if(ptr_classifier != NULL){
        delete ptr_classifier;
        ptr_classifier = NULL;
    }
}

int Classifier_interface::predict(const arma::colvec& x){
   if(ptr_classifier != NULL){
       return ptr_classifier->predict(x);
   }else{
       return -1;
   }
}

void Classifier_interface::get_probability(arma::colvec& prob){
    if(ptr_classifier != NULL){
        prob = ptr_classifier->get_probabilities();
    }
}


}
